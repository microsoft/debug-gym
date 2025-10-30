import json
from unittest.mock import patch

from debug_gym.agents.history_tracker import HistoryTracker, build_history_prompt
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.llms import OpenAILLM
from debug_gym.llms.base import LLMConfigRegistry, LLMResponse


def test_history_tracker(build_env_info):
    ht = HistoryTracker(history_steps=3)

    # should start empty
    assert len(ht) == 0
    assert ht.get() == ([], [])
    assert ht.get_all() == []
    assert ht.score() == 0
    assert ht.prompt_response_pairs == []

    # json should return an empty dict
    assert ht.json() == {}

    # prepare some data
    tool_2 = ToolCall(id="2", name="action2", arguments={"a2_args": "a2_args"})
    tool_3 = ToolCall(id="3", name="action3", arguments={})
    tool_4 = ToolCall(id="4", name="action4", arguments={"a4_args": "a4_args"})
    tool_5 = ToolCall(id="5", name="action5", arguments={})
    action_content_2 = "content_2_1"
    action_content_3 = "content_3_2"
    action_content_4 = "content_4_1"
    action_content_5 = "content_5_2"
    action_reasoning_2 = "reasoning_2_1"
    action_reasoning_3 = "reasoning_3_2"
    action_reasoning_4 = "reasoning_4_1"
    action_reasoning_5 = "reasoning_5_2"
    env_info_1 = build_env_info(
        step_observation="obs1",
        action_tool_call=None,
        action_reasoning=None,
        action_content=None,
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
    ht.step(env_info_1, None)
    ht.step(env_info_2, [llm_response_2])
    ht.step(env_info_3, [llm_response_3])
    ht.step(env_info_4, [llm_response_4])
    ht.step(env_info_5, [llm_response_5])

    # get_all should return all steps
    assert ht.get_all() == [env_info_1, env_info_2, env_info_3, env_info_4, env_info_5]

    # get should return the last 3 steps
    env_infos, llm_responses = ht.get()
    assert env_infos == [env_info_3, env_info_4, env_info_5]
    assert llm_responses == [[llm_response_3], [llm_response_4], [llm_response_5]]

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
        "obs": "obs1",
        "prompt_response_pairs": None,
        "rewrite_consumed": 0,
    }

    # score should return the sum of the scores
    assert ht.score() == 15

    # len should return the number of steps
    assert len(ht) == 5

    # Test cloning
    ht_clone = ht.clone()
    assert ht_clone.memory == ht.memory
    assert ht_clone.prompt_response_pairs == ht.prompt_response_pairs
    assert ht_clone.history_steps == ht.history_steps
    assert ht_clone is not ht

    # should reset properly
    ht.reset()
    assert len(ht) == 0
    assert ht.get() == ([], [])
    assert ht.get_all() == []
    assert ht.score() == 0
    assert ht.prompt_response_pairs == []

    # json should return an empty dict
    assert ht.json() == {}


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "openai": {
                "model": "openai",
                "tokenizer": "gpt-4o",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "api_version": "v1",
                "tags": ["azure openai"],
            }
        }
    ),
)
def test_build_history_prompt(mock_llm_config, build_env_info):
    # test with empty history
    ht = HistoryTracker(history_steps=3)
    llm = OpenAILLM("openai")
    messages = build_history_prompt(ht, llm)
    expected = []
    assert messages == expected

    # test with non-empty history
    ht = HistoryTracker(history_steps=3)
    # prepare some data
    env_info_1 = build_env_info(
        step_observation="obs1",
        action_tool_call=None,
        action_reasoning=None,
        action_content=None,
        score=1,
        rewrite_counter=0,
    )

    action_1 = ToolCall(id="1", name="action1", arguments={"a1_args": "a1_args"})
    llm_response_1 = LLMResponse(
        prompt="prompt_1", response="response_1", tool=action_1
    )
    env_info_2 = build_env_info(
        step_observation="obs2",
        action_reasoning="response_1",
        action_content="response_1",
        action_tool_call=action_1,
        score=2,
        rewrite_counter=0,
    )

    action_2 = ToolCall(id="2", name="action2", arguments={})
    llm_response_2 = LLMResponse(
        prompt="prompt_2", response="response_2", tool=action_2
    )
    env_info_3 = build_env_info(
        step_observation="obs3",
        action_reasoning="response_2",
        action_content="response_2",
        action_tool_call=action_2,
        score=3,
        rewrite_counter=0,
    )

    action_3 = ToolCall(id="3", name="action3", arguments={})
    llm_response_3 = LLMResponse(
        prompt="prompt_3", response="response_3", tool=action_3
    )
    env_info_4 = build_env_info(
        step_observation="obs4",
        action_reasoning="response_3",
        action_content="response_3",
        action_tool_call=action_3,
        score=4,
        rewrite_counter=1,
    )

    action_4 = ToolCall(id="4", name="action4", arguments={"a4_args": "a4_args"})
    llm_response_4 = LLMResponse(
        prompt="prompt_4", response="response_4", tool=action_4
    )
    env_info_5 = build_env_info(
        step_observation="obs5",
        action_reasoning="response_4",
        action_content="response_4",
        action_tool_call=action_4,
        score=5,
        rewrite_counter=1,
    )

    # push some steps
    ht.step(env_info_1, None)
    ht.step(env_info_2, [llm_response_1])
    ht.step(env_info_3, [llm_response_2])
    ht.step(env_info_4, [llm_response_3])
    ht.step(env_info_5, [llm_response_4])

    # reset_prompt_history_after_rewrite is False
    messages = build_history_prompt(ht, llm, reset_prompt_history_after_rewrite=False)
    expected = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "2",
                    "function": {
                        "name": "action2",
                        "arguments": json.dumps({}),
                    },
                },
            ],
            "content": "response_2",
        },
        {
            "role": "tool",
            "tool_call_id": "2",
            "name": "action2",
            "content": "obs3",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "3",
                    "function": {
                        "name": "action3",
                        "arguments": json.dumps({}),
                    },
                },
            ],
            "content": "response_3",
        },
        {
            "role": "tool",
            "tool_call_id": "3",
            "name": "action3",
            "content": "obs4",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "4",
                    "function": {
                        "name": "action4",
                        "arguments": json.dumps({"a4_args": "a4_args"}),
                    },
                },
            ],
            "content": "response_4",
        },
        {
            "role": "tool",
            "tool_call_id": "4",
            "name": "action4",
            "content": "obs5",
        },
    ]

    assert messages == expected

    # reset_prompt_history_after_rewrite is True
    messages = build_history_prompt(ht, llm, reset_prompt_history_after_rewrite=True)
    expected = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "3",
                    "function": {
                        "name": "action3",
                        "arguments": json.dumps({}),
                    },
                },
            ],
            "content": "response_3",
        },
        {
            "role": "tool",
            "tool_call_id": "3",
            "name": "action3",
            "content": "obs4",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "4",
                    "function": {
                        "name": "action4",
                        "arguments": json.dumps({"a4_args": "a4_args"}),
                    },
                },
            ],
            "content": "response_4",
        },
        {
            "role": "tool",
            "tool_call_id": "4",
            "name": "action4",
            "content": "obs5",
        },
    ]
    assert messages == expected
