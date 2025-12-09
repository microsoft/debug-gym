import argparse
import json
import logging
import signal
from pathlib import Path

from debug_gym.agents.base_agent import create_agent
from debug_gym.experiment import add_tools, create_env
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.llms.base import LLM, LLMResponse
from debug_gym.llms.human import Human
from debug_gym.llms.utils import print_messages
from debug_gym.logger import DebugGymLogger


class AgentTimeoutException(BaseException):
    """Custom exception to handle timeouts in agent
    execution. Inherits from BaseException to ensure
    it is not caught by agent exception handling."""

    pass


def set_signal(timeout_seconds):
    """Set a signal handler for timeouts.
    Only works on Unix-like systems."""

    def timeout_handler(signum, frame):
        """Signal handler for timeout."""
        raise AgentTimeoutException

    if timeout_seconds > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)


def run_replay_agent(agent, task_name=None, args=None):
    step = 0
    info = None
    max_steps = agent.config["max_steps"]
    try:
        agent.history.reset()
        info = agent.env.reset(options={"task_name": task_name})
        # initial state does not have prompt and response
        agent.history.step(info, None)

        if info.resolved is True:
            agent.logger.report_progress(
                problem_id=task_name,
                step=1,
                total_steps=1,
                score=info.score,
                max_score=info.max_score,
                status="resolved",
            )
            return True

        agent.logger.info(
            "Available tools (in LLM's tool calling format):\n"
            f"{json.dumps(agent.llm.define_tools(info.tools), indent=4)}\n"
        )

        highscore = info.score
        for step in range(max_steps):
            agent.logger.info(f"\n{'='*20} STEP {step+1} {'='*20}\n")
            highscore = max(highscore, info.score)
            msg = f"[{task_name[:10]:<10}] Step {step} | Score: {info.score}/{info.max_score or '-'} [Best: {highscore}]"
            agent.logger.info(msg)

            messages = agent.build_prompt(info)
            if step < agent.config["replay_from"]:
                history_step = agent.config["trajectory"][step + 1]
                assert len(history_step["prompt_response_pairs"]) == 1
                prompt_response = history_step["prompt_response_pairs"][0]
                llm_response = LLMResponse(
                    prompt=prompt_response["prompt"],
                    response=prompt_response.get("response"),
                    reasoning_response=prompt_response.get(
                        "reasoning_response"
                    ),  # TODO: check if that's correct
                    tool=ToolCall(**history_step["action"]),
                    prompt_token_count=prompt_response["token_usage"]["prompt"],
                    response_token_count=prompt_response["token_usage"]["response"],
                )
                print_messages(messages, agent.logger)
                agent.logger.info(
                    f"LLM response - reasoning: {llm_response.reasoning_response}\n"
                    f"LLM response - content: {llm_response.response}\n"
                    f"LLM response - tool call: {llm_response.tool}"
                )
            else:
                llm_response = agent.llm(messages, info.tools)

            if args.debug and (args.debug_at is None or step >= args.debug_at):
                breakpoint()

            info = agent.env.step(
                llm_response.tool,
                llm_response.response,
                llm_response.reasoning_response,
            )
            agent.history.step(info, llm_response)

            if info.terminated or info.edit_counter >= agent.config["max_edit_steps"]:
                reason = "terminated" if info.resolved else "max_edit_steps reached"
                agent.logger.info(
                    f"Step: {step} | Score: {info.score}/{info.max_score if info.max_score else '-'} | Reason: {reason}"
                )
                # early stop, set current step and total steps to be the same
                agent.logger.report_progress(
                    problem_id=task_name,
                    step=step + 1,
                    total_steps=step + 1,
                    score=info.score,
                    max_score=info.max_score,
                    status="resolved" if info.resolved else "unresolved",
                )
                break
            # keep progress bar running until max_steps is reached
            agent.logger.report_progress(
                problem_id=task_name,
                step=step + 1,
                total_steps=max_steps + 1,
                score=info.score,
                max_score=info.max_score,
                status="running",
            )
        # max_steps was reached, task was either resolved or unresolved
        agent.logger.report_progress(
            problem_id=task_name,
            step=step + 1,
            total_steps=step + 1,
            score=info.score,
            max_score=info.max_score,
            status="resolved" if info.resolved else "unresolved",
        )
        return info.resolved
    except Exception:
        # report any error that happens during the run
        agent.logger.report_progress(
            problem_id=task_name,
            step=step + 1,
            total_steps=step + 1,
            score=info.score if info else 0,
            max_score=info.max_score if info else 1,
            status="error",
        )
        raise


def run_task(args, problem, config):
    set_signal(args.timeout)
    success = True
    env = None

    # Flag to not report errors from the agent, since they report
    # errors themselves and we want to avoid double reporting.
    report_progress_error = True

    exp_path = Path(config["output_path"]) / config["uuid"]
    task_logger = DebugGymLogger(
        problem,
        log_dir=exp_path,
        level=args.logging_level,
        mode="a",
    )

    try:
        task_logger.report_progress(
            problem_id=problem,
            step=0,
            total_steps=1,
            score=0,
            max_score=1,
            status="running",
        )

        env = create_env(config, task_logger)
        add_tools(env, config, task_logger)

        llm = LLM.instantiate(
            llm_name=config["llm_name"],
            llm_config_file_path=config.get("llm_config_file_path"),
            logger=task_logger,
        )

        agent = create_agent(
            config["agent_type"],
            config=config,
            env=env,
            llm=llm,
            logger=task_logger,
        )

        try:
            success = run_replay_agent(agent, task_name=problem, args=args)
        except KeyboardInterrupt:
            task_logger.error("Agent run was interrupted by user.")
            task_logger.report_progress(
                problem_id=problem,
                step=1,
                total_steps=1,
                score=0,
                max_score=1,
                status="error",
            )
            success = False
            raise
        except AgentTimeoutException:
            task_logger.error(
                f"Timeout: Problem `{problem}` exceeded "
                f"the time limit of {args.timeout} seconds."
            )
            task_logger.report_progress(
                problem_id=problem,
                step=1,
                total_steps=1,
                score=0,
                max_score=1,
                status="error",
            )
            success = False
            raise
        except:
            report_progress_error = False
            raise

        # optionally apply patch
        if config["save_patch"]:
            agent.save_patch(task_name=problem)

        # save log
        agent.log(task_name=problem)

    except Exception as e:
        task_logger.error(
            f"Task Error: {problem} - {e!r}. Run with --very-verbose "
            f"or check {task_logger.log_file} for more information."
        )
        task_logger.debug(
            f"Task {problem} generated an exception: {e!r}", exc_info=True
        )
        if report_progress_error:
            task_logger.report_progress(
                problem_id=problem,
                step=1,
                total_steps=1,
                score=0,
                max_score=1,
                status="error",
            )
        if args.debug:
            raise e

        success = False
    finally:
        # Close env and cancel any pending alarm
        signal.alarm(0)
        if env:
            env.close()
    return success


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay DebugGym experiment trajectory."
    )
    parser.add_argument(
        "trajectory_file",
        type=str,
        help="Path to the trajectory debug_gym.jsonl file to replay.",
    )
    parser.add_argument(
        "--replay-from",
        type=int,
        default=-1,
        help="Step index to start replaying from. Defaults to continuing from the end of the trajectory.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help="LLM model to use for replay.",
    )
    parser.add_argument(
        "--tools",
        type=json.loads,
        default=None,
        help='List of tools to enable, e.g. \'["pdb","grep","eval"]\'',
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Timeout in seconds for each problem (0 means no timeout).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        dest="logging_level",
        action="store_const",
        const=logging.INFO,
        help="Verbose mode",
        default=logging.WARNING,
    )
    group.add_argument(
        "-vv",
        "--very-verbose",
        dest="logging_level",
        action="store_const",
        const=logging.DEBUG,
        help="Verbose mode",
        default=logging.WARNING,
    )
    group.add_argument(
        "--logging-level",
        dest="logging_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--debug-at", type=int, default=None, help="Step index to enable debug mode at."
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    logger = DebugGymLogger("debug-gym", level=args.logging_level)
    logger.info(f"Loading trajectory from {args.trajectory_file}")

    # Load trajectory file and validate it.
    with open(args.trajectory_file, "r") as f:
        trajectory = json.loads(f.read())

    config = trajectory["config"]

    if args.debug_at is not None:
        args.debug = True

    # Make sure we are replaying from an existing step.
    if args.replay_from == -1:
        args.replay_from = len(trajectory["log"]) - 1

    if args.replay_from >= len(trajectory["log"]):
        raise ValueError(
            f"replay_from {args.replay_from} is greater than the number of steps in the trajectory {len(trajectory['log'])}."
        )

    config["trajectory"] = trajectory["log"]
    config["replay_from"] = args.replay_from
    config["llm_name"] = args.llm or config["llm_name"]
    config["output_path"] = str(
        Path(args.trajectory_file).parent / str(config["replay_from"])
    )
    config["uuid"] = config["llm_name"]

    # Create the environment to get the list of problems to run.
    env = create_env(config, logger=logger)
    problems = sorted(env.dataset)
    assert (
        len(problems) == 1
    ), "Replay only supports a single problem in the trajectory file."

    llm = LLM.instantiate(
        llm_name=config["llm_name"],
        llm_config_file_path=config.get("llm_config_file_path"),
        logger=logger,
    )

    # Stop live progress display if in Human mode (avoid conflicts with prompt_toolkit)
    if isinstance(llm, Human) or args.debug:
        logger.set_no_live()

    with logger.rich_progress(problems):
        for problem in problems:
            try:
                success = run_task(args, problem, config)
            except AgentTimeoutException:
                pass  # Handleled in run_agent, just continue
            except (KeyboardInterrupt, Exception) as e:
                raise e


if __name__ == "__main__":
    main()
