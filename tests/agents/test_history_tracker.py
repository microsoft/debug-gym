from debug_gym.agents.history_tracker import HistoryTracker, build_history_prompt
from debug_gym.agents.llm_api import LLMResponse


def test_history_tracker(build_env_info):
    ht = HistoryTracker(history_steps=3)

    # should start empty
    assert len(ht) == 0
    assert ht.get() == []
    assert ht.get_all() == []
    assert ht.score() == 0
    assert ht.prompt_response_pairs == []

    # json should return an empty dict
    assert ht.json() == {}

    # prepare some data
    env_info_1 = build_env_info(step_observation="obs1", action=None, score=1)
    env_info_2 = build_env_info(step_observation="obs2", action="action2", score=2)
    env_info_3 = build_env_info(step_observation="obs3", action="action3", score=3)
    env_info_4 = build_env_info(step_observation="obs4", action="action4", score=4)
    env_info_5 = build_env_info(step_observation="obs5", action="action5", score=5)

    # single prompt format
    llm_response_2 = LLMResponse("prompt_2_1", "response_2_1")
    # list of messages format
    llm_response_3 = LLMResponse(
        prompt=[
            {"role": "user", "content": "prompt_3_1"},
            {"role": "assistent", "content": "response_3_1"},
            {"role": "user", "content": "prompt_3_2"},
        ],
        response="response_3_2",
    )
    llm_response_4 = LLMResponse("prompt_4_1", "response_4_1", 4321, 1234)
    llm_response_5 = LLMResponse(
        prompt=[
            {"role": "user", "content": "prompt_5_1"},
            {"role": "assistent", "content": "response_5_1"},
            {"role": "user", "content": "prompt_5_2"},
        ],
        response="response_5_2",
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
    assert ht.get() == [env_info_3, env_info_4, env_info_5]

    # json should return the last step by default
    assert ht.json() == {
        "step_id": 4,
        "action": "action5",
        "obs": "obs5",
    }

    # json should return the speficied step
    assert ht.json(2) == {
        "step_id": 2,
        "action": "action3",
        "obs": "obs3",
    }

    # output token_usage if it exists
    assert ht.json(3, include_prompt_response_pairs=True) == {
        "step_id": 3,
        "action": "action4",
        "obs": "obs4",
        "prompt_response_pairs": [
            {
                "prompt": "prompt_4_1",
                "response": "response_4_1",
                "token_usage": {"prompt": 4321, "response": 1234},
            }
        ],
    }

    # json should return also the prompt-response pairs if include_prompt_response_pairs is True
    assert ht.json(2, include_prompt_response_pairs=True) == {
        "step_id": 2,
        "action": "action3",
        "obs": "obs3",
        "prompt_response_pairs": [
            {
                "prompt": [
                    {"role": "user", "content": "prompt_3_1"},
                    {"role": "assistent", "content": "response_3_1"},
                    {"role": "user", "content": "prompt_3_2"},
                ],
                "response": "response_3_2",
            }
        ],
    }

    # for 0-th step, prompt-response pairs should be None
    assert ht.json(0, include_prompt_response_pairs=True) == {
        "step_id": 0,
        "action": None,
        "obs": "obs1",
        "prompt_response_pairs": None,
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
        assert step.action not in ["action2", "action4"]
        assert step.action in [None, "action3", "action5"]

    # should reset properly
    ht.reset()
    assert len(ht) == 0
    assert ht.get() == []
    assert ht.get_all() == []
    assert ht.score() == 0
    assert ht.prompt_response_pairs == []

    # json should return an empty dict
    assert ht.json() == {}


def test_build_history_prompt(build_env_info):
    import json

    from debug_gym.gym.utils import unescape

    # test with empty history
    ht = HistoryTracker(history_steps=3)
    # use_conversational_prompt is False
    messages = build_history_prompt(ht, use_conversational_prompt=False)
    expected = [
        {"role": "user", "content": "No history of command and terminal outputs."}
    ]
    assert messages == expected
    # use_conversational_prompt is True
    messages = build_history_prompt(ht, use_conversational_prompt=True)
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
        step_observation="obs2", action="action2", score=2, rewrite_counter=0
    )
    env_info_3 = build_env_info(
        step_observation="obs3", action="action3", score=3, rewrite_counter=0
    )
    env_info_4 = build_env_info(
        step_observation="obs4", action="action4", score=4, rewrite_counter=1
    )
    env_info_5 = build_env_info(
        step_observation="obs5", action="action5", score=5, rewrite_counter=1
    )

    # push some steps
    ht.step(env_info_1)
    ht.step(env_info_2)
    ht.step(env_info_3)
    ht.step(env_info_4)
    ht.step(env_info_5)

    # use_conversational_prompt is False
    # reset_prompt_history_after_rewrite is False
    messages = build_history_prompt(
        ht, use_conversational_prompt=False, reset_prompt_history_after_rewrite=False
    )
    expected = [f"History of command and terminal outputs (the last 3 steps):"]
    history_messages = [
        {"step": 0, "command": "action3", "stdout": "obs3"},
        {"step": 1, "command": "action4", "stdout": "obs4"},
        {"step": 2, "command": "action5", "stdout": "obs5"},
    ]
    expected += ["\n" + unescape(json.dumps(history_messages, indent=4)) + "\n"]
    expected = [{"role": "user", "content": "\n".join(expected)}]
    assert messages == expected

    # reset_prompt_history_after_rewrite is True
    messages = build_history_prompt(
        ht, use_conversational_prompt=False, reset_prompt_history_after_rewrite=True
    )
    expected = [f"History of command and terminal outputs (the last 2 steps):"]
    history_messages = [
        {"step": 0, "command": "action4", "stdout": "obs4"},
        {"step": 1, "command": "action5", "stdout": "obs5"},
    ]
    expected += ["\n" + unescape(json.dumps(history_messages, indent=4)) + "\n"]
    expected = [{"role": "user", "content": "\n".join(expected)}]
    assert messages == expected

    # use_conversational_prompt is True
    # reset_prompt_history_after_rewrite is False
    messages = build_history_prompt(
        ht, use_conversational_prompt=True, reset_prompt_history_after_rewrite=False
    )
    expected = [
        {
            "role": "user",
            "content": "History of command and terminal outputs (the last 3 steps):",
        }
    ]
    history_messages = [
        {"role": "assistant", "content": "action3"},
        {"role": "user", "content": "obs3"},
        {"role": "assistant", "content": "action4"},
        {"role": "user", "content": "obs4"},
        {"role": "assistant", "content": "action5"},
        {"role": "user", "content": "obs5"},
    ]
    expected += history_messages
    assert messages == expected
    # reset_prompt_history_after_rewrite is True
    messages = build_history_prompt(
        ht, use_conversational_prompt=True, reset_prompt_history_after_rewrite=True
    )
    expected = [
        {
            "role": "user",
            "content": "History of command and terminal outputs (the last 2 steps):",
        }
    ]
    history_messages = [
        {"role": "assistant", "content": "action4"},
        {"role": "user", "content": "obs4"},
        {"role": "assistant", "content": "action5"},
        {"role": "user", "content": "obs5"},
    ]
    expected += history_messages
    assert messages == expected
