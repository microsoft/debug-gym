import argparse
import asyncio
import os
import time

from rich.progress import BarColumn, Progress, TextColumn
from termcolor import colored

from froggy.agents.llm_api import instantiate_llm, postprocess_messages, print_messages
from froggy.envs import AiderBenchmarkEnv
from froggy.tools.patchers import Patcher
from froggy.tools.pdb import PDBTool
from froggy.utils import HistoryTracker


class MarkdownLogger:
    def __init__(self, output_path):
        self.output_path = output_path
        self.previous_step = 0

    def _color_message(self, message, header):
        color = "black"  # default color

        if header == "user":
            color = "#007b7b"  # or use "blue" as per your preference
        elif header == "assistant":
            color = "#006400"
        elif header == "system":
            color = "#b8860b"

        escaped_message = message.replace("#", "\\#")
        return f'<span style="color:{color};"> {escaped_message} </span>'

    def _color_message(self, message, header):
        color = "black"  # default color

        if header == "user":
            color = "#007b7b"  # dark cyan
        elif header == "assistant":
            color = "#006400"  # dark green
        elif header == "system":
            color = "#b8860b"  # dark goldenrod

        lines = message.splitlines()
        colored_lines = []
        inside_code_block = False
        writing_program = False

        for line in lines:
            if line.strip().startswith("```"):
                # Toggle code block mode
                inside_code_block = not inside_code_block
                colored_lines.append(line)  # Don't add color to ``` line
            elif line.strip().startswith("Current program:"):
                writing_program = True
                colored_lines.append(line)
                colored_lines.append("```")
            elif writing_program or inside_code_block or line.strip().startswith("#"):
                # No color inside code blocks
                colored_lines.append(line)
            else:
                # Apply color outside code blocks
                colored_lines.append(f'<span style="color:{color};">{line}</span>')

        if writing_program:
            colored_lines.append("```")

        return "\n".join(colored_lines)

    def log(self, message, header, step):
        # add 2 carriage returns to the message

        colored_message = self._color_message(message, header)
        with open(self.output_path, "a") as f:
            if step != self.previous_step:
                f.write("<br><br>\n")  # HTML line breaks for extra spacing
                self.previous_step = step

            if step is not None:
                f.write(f"# Step {step} ***** {header.capitalize()} ***** \n")
            else:
                f.write(f"# {header.capitalize()} ***** \n")

            f.write(f"{colored_message}\n\n")


def compose_ingame_prompt_pdb_as_user(info, question, history=None):
    # system prompt -- start
    messages = [
        {
            "role": "system",
            "content": "Your are a pro Python programmer. You are debugging a Python program. Your goal is to use the pdb tool to investigate the code, you can set breakpoints and print necessary values to help you identify the bugs. After that, fix the bugs by rewriting a few lines of the code.",
        },
    ]

    if len(info["instructions"]) > 0:
        messages[-1]["content"] += f"\n{info['instructions']}"

    messages[-1]["content"] += f"\n{info['current_code_with_line_number']}\n"
    # system prompt -- end

    if history:

        def _build_history(history):
            # only consider up to the last rewrite!
            start = 0

            for i, info in enumerate(history.memory):
                if info["is_rewrite"]:
                    start = i

            messages = []
            infos = history.memory[start:]

            for info in infos:
                # do not append rewrites the agent has done, this is already encapsulated in
                # the current code in the system prompt
                if info["action"] is not None and not info["is_rewrite"]:
                    messages.append(
                        {"role": "assistant", "content": f"{info['action']}"}
                    )

                obs = f"{info['obs']}"
                if (
                    messages
                    and messages[-1]["role"] == "assistant"
                    and obs.startswith(messages[-1]["content"])
                ):
                    obs = obs[len(messages[-1]["content"]) :].strip()
                messages.append({"role": "user", "content": obs})

            return messages

        messages.extend(_build_history(history))

    messages.append({"role": "user", "content": question})
    return messages


async def run(
    progress,
    task_name,
    args,
    llm,
):
    nb_steps = args.nb_steps
    nb_trials = args.nb_trials
    nb_rewrites = args.nb_rewrites
    interactive = args.interactive

    history = HistoryTracker(nb_steps)

    env = AiderBenchmarkEnv(output_dir=args.output)
    env.add_tool(Patcher.get(args.patch_type))
    if args.llm_agent == "froggy":
        env.add_tool(PDBTool())
    env.seed(args.seed)

    questions, temp_per_question = get_questions_and_temps(args, env)
    _, info = env.reset(task_name=task_name)
    history.step(info)

    reward = 0
    step_id = 0
    rewrite_id = 0
    traces = {}
    done = False

    task_id = progress.add_task(
        f"[green]{task_name}", total=nb_steps * nb_trials, score=0
    )

    logger = MarkdownLogger(f"{args.output}/{task_name}.md")

    while not info["done"]:
        question = questions[0] if step_id == 0 else questions[-1]
        temperature = temp_per_question[0] if step_id == 0 else temp_per_question[-1]

        messages = compose_ingame_prompt_pdb_as_user(info, question, history=history)
        messages = postprocess_messages(messages, llm.context_length)

        if args.verbose:
            print(colored(f"------- START OF TURN {step_id} --------\n", "red"))
            print_messages(messages)

        for msg in messages:
            logger.log(msg["content"], msg["role"], step_id)

        answer = None
        if interactive:
            answer = input("> ").strip()

        if not answer:
            answer = await llm(messages, info, temperature=temperature)

        if args.verbose:
            print("Agent action: " + colored(answer, "green"))

        logger.log(answer, "assistant", step_id)

        if args.debug:
            progress.stop()
            breakpoint()
            progress.start()

        _, reward, done, info = env.step(answer)
        history.step(info)

        progress.update(task_id, advance=1, score=info["score"] / info["max_score"])

        rewrite_id += info["is_rewrite"]
        step_id += 1

        if args.nb_rewrites is not None and rewrite_id >= args.nb_rewrites:
            break

        if step_id >= nb_steps:
            break

    progress.update(task_id, advance=nb_steps, score=info["score"] / info["max_score"])
    time.sleep(0.1)
    progress.remove_task(task_id)

    return {
        "score": info["score"] / info["max_score"],
        "history": history,
    }


def get_questions_and_temps(args, env):
    # hacky
    assert "```rewrite" in env.actions_str, "Needs at least a patcher tool!"

    # The first question in the list is the one used to write the initial code, and the second question is the one used to either continue debugging or to fix the code
    if args.llm_agent == "baseline":
        # the baseline agents has two questions, one for writing initial code and one for fixing it
        questions = [
            "Based on the history information, rewrite the code to fix the test using solely the ```rewrite command. NOTHING ELSE.",
            "Based on the history information, rewrite the code to fix the test using solely the ```rewrite command. NOTHING ELSE.",
        ]
    elif args.llm_agent == "froggy":
        # the other agents have two questions, one for writing initial code and one for either debugging or fixing it
        questions = [
            "Based on the history information, rewrite the code to fix the test using solely the ```rewrite tool. NOTHING ELSE.",
            f"Based on the history information, continue to acquire information using the tools at your disposal until you are ready to rewrite the code! Available tools: {env.actions_str}. Only output the tool command. NOTHING ELSE.",
        ]
    else:
        raise ValueError("Invalid LLM agent.")

    temp_per_question = [0.3, 0.3]
    return questions, temp_per_question


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", help="Name of the LLM", default="random")
    parser.add_argument(
        "--llm_agent", help="Name of the LLM Agent to use", default=None
    )
    parser.add_argument("--nb-steps", help="Number of steps", default=1, type=int)
    parser.add_argument("--nb-trials", help="Number of trials", default=1, type=int)
    parser.add_argument(
        "--tasks", nargs="+", help="List of task ID to evaluate", default="all"
    )
    parser.add_argument(
        "--seed",
        help="Control which sample to pick on env.reset(). Default: %(default)s",
        default=0,
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--nb-rewrites", default=None, type=int, help="Number of maximum rewrites"
    )
    parser.add_argument("--num-workers", default=1, type=int, help="Num workers")
    parser.add_argument(
        "--resume", action="store_true", help="Resume experiment with scores in json."
    )
    parser.add_argument(
        "--patch-type",
        type=str,
        default="custom",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        help="Path where to generated code. Default: %(default)s",
        default="data/output",
    )

    args = parser.parse_args()

    llm = instantiate_llm(args, use_async=True)

    score_dict = {}
    os.makedirs(args.output, exist_ok=True)

    # only to get all the tasks
    env = AiderBenchmarkEnv(output_dir=args.output)
    env.seed(args.seed)

    if args.tasks == "all":
        args.tasks = list(env.dataset)

    if args.debug:
        args.tasks = args.tasks[:1]

    print("Testing on:", args.tasks)
    if args.resume:
        # be more robust here and check if the file exists
        if not os.path.exists(f"{args.output}/scores.json"):
            print("!!! No previous evaluation found, starting from scratch... !!!")
        else:
            with open(f"{args.output}/scores.json", "r") as f:
                import json

                score_dict = json.load(f)

            # only loop over tasks that have not been evaluated yet
            args.tasks = [task for task in args.tasks if task not in score_dict.keys()]
            # print something nice
            print("!!! Resuming from previous evaluation... !!!")

    if args.llm_agent == "baseline":
        print(f"Running baseline agent with {args.nb_steps} steps.")

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("[green]Score: {task.fields[score]}"),
        disable=args.verbose,
    )
    main_task = progress.add_task(
        "[blue]Overall Progress", total=len(args.tasks), score=0
    )

    tasks = []
    for task in args.tasks:
        tasks.append(
            run(
                progress,
                task,
                args,
                llm,
            )
        )

    with progress:
        for batch in range(0, len(tasks), args.num_workers):
            scores = await asyncio.gather(*tasks[batch : batch + args.num_workers])

            for i, score in enumerate(scores):
                score_dict[args.tasks[batch + i]] = score["score"]
                score["history"].save(
                    f"{args.output}/{args.tasks[batch + i]}_history.json"
                )

            score_dict["mean"] = sum(score_dict.values()) / len(score_dict)

            with open(f"{args.output}/scores.json", "w") as f:
                import json

                json.dump(score_dict, f, indent=4)

            progress.update(main_task, advance=len(scores), score=score_dict["mean"])

    print("Normalized scores on each coding task:", score_dict)
    print("Mean score:", score_dict["mean"])


if __name__ == "__main__":
    asyncio.run(main())
