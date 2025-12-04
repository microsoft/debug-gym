from unittest.mock import patch

from debug_gym.agents.history_tracker import HistoryTracker
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.llms import OpenAILLM
from debug_gym.llms.base import LLMConfigRegistry, LLMResponse


def test_history_tracker(build_env_info):
    ht = HistoryTracker()

    # should start empty
    assert len(ht) == 0
    assert ht.get() == ([], [])
    assert ht.score() == 0

    # json should return an empty dict
    assert ht.json() == {}

    # prepare some data
    tool_1 = ToolCall(id="1", name="action1", arguments={"a1_args": "a1_args"})
    tool_2 = ToolCall(id="2", name="action2", arguments={"a2_args": "a2_args"})
    tool_3 = ToolCall(id="3", name="action3", arguments={})
    tool_4 = ToolCall(id="4", name="action4", arguments={"a4_args": "a4_args"})
    tool_5 = ToolCall(id="5", name="action5", arguments={})
    action_content_1 = "content_1_1"
    action_content_2 = "content_2_1"
    action_content_3 = "content_3_2"
    action_content_4 = "content_4_1"
    action_content_5 = "content_5_2"
    action_reasoning_1 = "reasoning_1_1"
    action_reasoning_2 = "reasoning_2_1"
    action_reasoning_3 = "reasoning_3_2"
    action_reasoning_4 = "reasoning_4_1"
    action_reasoning_5 = "reasoning_5_2"
    env_info_0 = build_env_info(
        step_observation="initial_obs",
        action_tool_call=None,
        action_reasoning=None,
        action_content=None,
        score=0,
        rewrite_counter=0,
    )
    env_info_1 = build_env_info(
        step_observation="obs1",
        action_tool_call=tool_1,
        action_reasoning=action_reasoning_1,
        action_content=action_content_1,
        score=1,
        rewrite_counter=0,
    )
    env_info_2 = build_env_info(
        step_observation="obs2",
        action_tool_call=tool_2,
        action_reasoning=action_reasoning_2,
        action_content=action_content_2,
        score=2,
        rewrite_counter=0,
    )
    env_info_3 = build_env_info(
        step_observation="obs3",
        action_tool_call=tool_3,
        action_reasoning=action_reasoning_3,
        action_content=action_content_3,
        score=3,
        rewrite_counter=1,
    )
    env_info_4 = build_env_info(
        step_observation="obs4",
        action_tool_call=tool_4,
        action_reasoning=action_reasoning_4,
        action_content=action_content_4,
        score=4,
        rewrite_counter=1,
    )
    env_info_5 = build_env_info(
        step_observation="obs5",
        action_tool_call=tool_5,
        action_reasoning=action_reasoning_5,
        action_content=action_content_5,
        score=5,
        rewrite_counter=2,
    )

    # single prompt format
    llm_response_1 = LLMResponse("prompt_1_1", "response_1_1", tool=tool_1)
    llm_response_2 = LLMResponse("prompt_2_1", "response_2_1", tool=tool_2)
    # list of messages format
    llm_response_3 = LLMResponse(
        prompt=[
            {"role": "user", "content": "prompt_3_1"},
            {"role": "assistent", "content": "response_3_1"},
            {"role": "user", "content": "prompt_3_2"},
        ],
        response="content_3_2",
        reasoning_response="reasoning_3_2",
        tool=tool_3,
    )
    llm_response_4 = LLMResponse(
        "prompt_4_1",
        "response_4_1",
        tool=tool_4,
        prompt_token_count=4321,
        response_token_count=1234,
    )
    llm_response_5 = LLMResponse(
        prompt=[
            {"role": "user", "content": "prompt_5_1"},
            {"role": "assistent", "content": "response_5_1"},
            {"role": "user", "content": "prompt_5_2"},
        ],
        response="content_5_2",
        reasoning_response="reasoning_5_2",
        tool=tool_5,
    )

    # push some steps and prompt-response pairs
    # at 0-th step, there is no prompt-response pair
    ht.init(None, None, env_info_0)
    ht.step(env_info_1, llm_response_1)
    ht.step(env_info_2, llm_response_2)
    ht.step(env_info_3, llm_response_3)
    ht.step(env_info_4, llm_response_4)
    ht.step(env_info_5, llm_response_5)

    # check initial state
    assert ht.env_initial_observation == env_info_0
    # get should return all steps except initial state
    env_infos, llm_responses = ht.get()
    assert env_infos == [env_info_1, env_info_2, env_info_3, env_info_4, env_info_5]
    assert llm_responses == [
        llm_response_1,
        llm_response_2,
        llm_response_3,
        llm_response_4,
        llm_response_5,
    ]

    # json should return the last step by default
    assert ht.json() == {
        "step_id": 4,
        "reasoning": action_reasoning_5,
        "content": action_content_5,
        "action": {"id": "5", "name": "action5", "arguments": {}},
        "obs": "obs5",
        "rewrite_consumed": 2,
        "prompt_response_pairs": [
            {
                "prompt": [
                    {"role": "user", "content": "prompt_5_1"},
                    {"role": "assistent", "content": "response_5_1"},
                    {"role": "user", "content": "prompt_5_2"},
                ],
                "response": "content_5_2",
                "reasoning_response": "reasoning_5_2",
                "tool": {"id": "5", "name": "action5", "arguments": {}},
            }
        ],
    }

    # json should return the speficied step
    assert ht.json(2) == {
        "step_id": 2,
        "reasoning": action_reasoning_3,
        "content": action_content_3,
        "action": {"id": "3", "name": "action3", "arguments": {}},
        "obs": "obs3",
        "rewrite_consumed": 1,
        "prompt_response_pairs": [
            {
                "prompt": [
                    {"role": "user", "content": "prompt_3_1"},
                    {"role": "assistent", "content": "response_3_1"},
                    {"role": "user", "content": "prompt_3_2"},
                ],
                "response": "content_3_2",
                "reasoning_response": "reasoning_3_2",
                "tool": {"id": "3", "name": "action3", "arguments": {}},
            }
        ],
    }

    # output token_usage if it exists
    assert ht.json(3) == {
        "step_id": 3,
        "reasoning": action_reasoning_4,
        "content": action_content_4,
        "action": {"id": "4", "name": "action4", "arguments": {"a4_args": "a4_args"}},
        "obs": "obs4",
        "prompt_response_pairs": [
            {
                "prompt": "prompt_4_1",
                "response": "response_4_1",
                "tool": {
                    "id": "4",
                    "name": "action4",
                    "arguments": {"a4_args": "a4_args"},
                },
                "token_usage": {"prompt": 4321, "response": 1234},
            }
        ],
        "rewrite_consumed": 1,
    }

    # for 0-th step, prompt-response pairs should be None
    assert ht.json(0) == {
        "step_id": 0,
        "reasoning": None,
        "content": None,
        "action": None,
        "obs": "initial_obs",
        "system_message": None,
        "problem_message": None,
        "prompt_response_pairs": None,
        "rewrite_consumed": 0,
    }

    # score should return the sum of the scores
    assert ht.score() == 15

    # len should return the number of steps
    assert len(ht) == 5

    # Test cloning
    ht_clone = ht.clone()
    assert ht_clone.env_observations == ht.env_observations
    assert ht_clone is not ht

    # should reset properly
    ht.reset()
    assert len(ht) == 0
    assert ht.get() == ([], [])
    assert ht.score() == 0

    # json should return an empty dict
    assert ht.json() == {}
