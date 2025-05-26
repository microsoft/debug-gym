from debug_gym.agents.history_tracker import HistoryTracker, build_history_prompt
from debug_gym.agents.llm_api import LLMResponse
from debug_gym.gym.tools.tool import ToolCall


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
    env_info_1 = build_env_info(
        step_observation="obs1", action=None, score=1, rewrite_counter=0
    )
    env_info_2 = build_env_info(
        step_observation="obs2", action=tool_2, score=2, rewrite_counter=0
    )
    env_info_3 = build_env_info(
        step_observation="obs3", action=tool_3, score=3, rewrite_counter=1
    )
    env_info_4 = build_env_info(
        step_observation="obs4", action=tool_4, score=4, rewrite_counter=1
    )
    env_info_5 = build_env_info(
        step_observation="obs5", action=tool_5, score=5, rewrite_counter=2
    )

    # single prompt format
    llm_response_2 = LLMResponse("prompt_2_1", "response_2_1", tool_2)
    # list of messages format
    llm_response_3 = LLMResponse(
        prompt=[
            {"role": "user", "content": "prompt_3_1"},
            {"role": "assistent", "content": "response_3_1"},
            {"role": "user", "content": "prompt_3_2"},
        ],
        response="response_3_2",
        tool=tool_3,
    )
    llm_response_4 = LLMResponse("prompt_4_1", "response_4_1", tool_4, 4321, 1234)
    llm_response_5 = LLMResponse(
        prompt=[
            {"role": "user", "content": "prompt_5_1"},
            {"role": "assistent", "content": "response_5_1"},
            {"role": "user", "content": "prompt_5_2"},
        ],
        response="response_5_2",
        tool=tool_5,
    )

    # push some steps and prompt-response pairs
    # at 0-th step, there is no prompt-response pair
    ht.step(env_info_1, None)
    ht.step(env_info_2, llm_response_2)
    ht.step(env_info_3, llm_response_3)
    ht.step(env_info_4, llm_response_4)
    ht.step(env_info_5, llm_response_5)

    # get_all should return all steps
    assert ht.get_all() == [env_info_1, env_info_2, env_info_3, env_info_4, env_info_5]

    # get should return the last 3 steps
    env_infos, llm_responses = ht.get()
    assert env_infos == [env_info_3, env_info_4, env_info_5]
    assert llm_responses == [[llm_response_3], [llm_response_4], [llm_response_5]]

    # json should return the last step by default
    assert ht.json() == {
        "step_id": 4,
        "action": {"id": "5", "name": "action5", "arguments": {}},
        "obs": "obs5",
        "rewrite_consumed": 2,
    }

    # json should return the speficied step
    assert ht.json(2) == {
        "step_id": 2,
        "action": {"id": "3", "name": "action3", "arguments": {}},
        "obs": "obs3",
        "rewrite_consumed": 1,
    }

    # output token_usage if it exists
    assert ht.json(3, include_prompt_response_pairs=True) == {
        "step_id": 3,
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

    # json should return also the prompt-response pairs if include_prompt_response_pairs is True
    assert ht.json(2, include_prompt_response_pairs=True) == {
        "step_id": 2,
        "action": {"id": "3", "name": "action3", "arguments": {}},
        "obs": "obs3",
        "prompt_response_pairs": [
            {
                "prompt": [
                    {"role": "user", "content": "prompt_3_1"},
                    {"role": "assistent", "content": "response_3_1"},
                    {"role": "user", "content": "prompt_3_2"},
                ],
                "response": "response_3_2",
                "tool": {"id": "3", "name": "action3", "arguments": {}},
            }
        ],
        "rewrite_consumed": 1,
    }

    # for 0-th step, prompt-response pairs should be None
    assert ht.json(0, include_prompt_response_pairs=True) == {
        "step_id": 0,
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

    # test filtering out
    ht_filtered = ht.filter_out(actions=["action2", "action4"])
    for step in ht_filtered.get_all():
        assert step.action not in [tool_2, tool_4]
        assert step.action in [None, tool_3, tool_5]

    # should reset properly
    ht.reset()
    assert len(ht) == 0
    assert ht.get() == ([], [])
    assert ht.get_all() == []
    assert ht.score() == 0
    assert ht.prompt_response_pairs == []

    # json should return an empty dict
    assert ht.json() == {}


def test_build_history_prompt(build_env_info, llm_mock):
    # test with empty history
    ht = HistoryTracker(history_steps=3)
    messages = build_history_prompt(ht, llm_mock)
    expected = [
        {"role": "user", "content": "No history of command and terminal outputs."}
    ]
    assert messages == expected

    # test with non-empty history
    ht = HistoryTracker(history_steps=3)
    # prepare some data
    env_info_1 = build_env_info(
        step_observation="obs1", action=None, score=1, rewrite_counter=0
    )
    env_info_2 = build_env_info(
        step_observation="obs2",
        action={"id": "2", "name": "action2", "arguments": {"a2_args": "a2_args"}},
        score=2,
        rewrite_counter=0,
    )
    env_info_3 = build_env_info(
        step_observation="obs3",
        action={"id": "3", "name": "action3", "arguments": {}},
        score=3,
        rewrite_counter=0,
    )
    env_info_4 = build_env_info(
        step_observation="obs4",
        action={"id": "4", "name": "action4", "arguments": {"a4_args": "a4_args"}},
        score=4,
        rewrite_counter=1,
    )
    env_info_5 = build_env_info(
        step_observation="obs5",
        action={"id": "5", "name": "action5", "arguments": {}},
        score=5,
        rewrite_counter=1,
    )

    # push some steps
    ht.step(env_info_1, None)
    ht.step(env_info_2, None)
    ht.step(env_info_3, None)
    ht.step(env_info_4, None)
    ht.step(env_info_5, None)

    # reset_prompt_history_after_rewrite is False
    messages = build_history_prompt(
        ht, llm_mock, reset_prompt_history_after_rewrite=False
    )

    expected = [
        {
            "role": "user",
            "content": "History of command and terminal outputs (the last 2 steps):",
        },
        {"role": "role", "content": {"id": "3", "name": "action3", "arguments": {}}},
        {
            "role": "role",
            "content": {
                "id": "4",
                "name": "action4",
                "arguments": {"a4_args": "a4_args"},
            },
        },
        {"role": "role", "content": {"id": "5", "name": "action5", "arguments": {}}},
    ]
    assert messages == expected

    # reset_prompt_history_after_rewrite is True
    messages = build_history_prompt(
        ht, llm_mock, reset_prompt_history_after_rewrite=True
    )
    expected = [
        {
            "role": "user",
            "content": "History of command and terminal outputs (the last 1 steps):",
        },
        {
            "role": "role",
            "content": {
                "id": "4",
                "name": "action4",
                "arguments": {"a4_args": "a4_args"},
            },
        },
        {"role": "role", "content": {"id": "5", "name": "action5", "arguments": {}}},
    ]
    assert messages == expected
