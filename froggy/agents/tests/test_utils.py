from unittest.mock import patch

import pytest

from froggy.agents.utils import (
    HistoryTracker,
    _build_history_prompt,
    load_config,
    trim_prompt_messages,
)


def test_trim_prompt_messages():
    def token_counter(text):
        return len(text)

    with pytest.raises(Exception, match="messages should not be empty"):
        trim_prompt_messages([], 5, token_counter)

    with pytest.raises(
        Exception,
        match='all messages should be dictionaries with keys "content" and "role"',
    ):
        messages = [{"role": "system", "key": "System message"}]
        trim_prompt_messages(messages, 20, token_counter)

    with pytest.raises(Exception, match="the last message should be from the user"):
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Assistant message"},
        ]
        trim_prompt_messages(messages, 20, token_counter)

    with pytest.raises(
        Exception,
        match="if two consecutive messages are from the same role, they should be merged first",
    ):
        messages = [
            {"role": "system", "content": "System message 1"},
            {"role": "system", "content": "System message 2"},
            {"role": "user", "content": "User message"},
        ]
        trim_prompt_messages(messages, 20, token_counter)

    with pytest.raises(Exception, match="context_length should be non-negative"):
        messages = [{"role": "user", "content": "User message"}]
        trim_prompt_messages(messages, -1, token_counter)

    messages = [{"role": "user", "content": "User message"}]
    assert trim_prompt_messages(messages, 0, token_counter) == messages

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
    ]
    expected = [{"role": "user", "content": "User message"}]
    assert trim_prompt_messages(messages, 20, token_counter) == expected

    messages = [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "User message 2"},
    ]
    expected = messages
    assert trim_prompt_messages(messages, 200, token_counter) == expected

    messages = [
        {"role": "user", "content": "User message 1"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "User message 2"},
    ]
    expected = [
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "User message 2"},
    ]
    assert trim_prompt_messages(messages, 35, token_counter) == expected

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "User message 2"},
    ]
    expected = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message 2"},
    ]
    assert trim_prompt_messages(messages, 35, token_counter) == expected

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message 1"},
        {"role": "assistant", "content": "Assistant message 1"},
        {"role": "user", "content": "User message 2"},
        {"role": "assistant", "content": "Assistant message 2"},
        {"role": "user", "content": "User message 3"},
        {"role": "assistant", "content": "Assistant message 3"},
        {"role": "user", "content": "User message 4"},
    ]
    expected = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message 3"},
        {"role": "assistant", "content": "Assistant message 3"},
        {"role": "user", "content": "User message 4"},
    ]
    assert trim_prompt_messages(messages, 65, token_counter) == expected


def test_history_tracker():
    ht = HistoryTracker(history_steps=3)

    # should start empty
    assert len(ht) == 0
    assert ht.get() == []
    assert ht.get_all() == []
    assert ht.score() == 0
    assert ht.prompt_response_pairs == [
        []
    ]  # at 0-th step, there is no prompt-response pair

    # json should return an empty dict
    assert ht.json() == {}

    # push some steps
    ht.step({"obs": "obs1", "action": None, "score": 1})
    ht.step({"obs": "obs2", "action": "action2", "score": 2})
    ht.step({"obs": "obs3", "action": "action3", "score": 3})
    ht.step({"obs": "obs4", "action": "action4", "score": 4, "token_usage": 12345})
    ht.step({"obs": "obs5", "action": "action5", "score": 5})
    # push some prompt-response pairs
    ht.save_prompt_response_pairs([("prompt_2_1", "response_2_1")])
    ht.save_prompt_response_pairs(
        [("prompt_3_1", "response_3_1"), ("prompt_3_2", "response_3_2")]
    )
    ht.save_prompt_response_pairs([("prompt_4_1", "response_4_1")])
    ht.save_prompt_response_pairs(
        [("prompt_5_1", "response_5_1"), ("prompt_5_2", "response_5_2")]
    )

    # get_all should return all steps
    assert ht.get_all() == [
        {"obs": "obs1", "action": None, "score": 1},
        {"obs": "obs2", "action": "action2", "score": 2},
        {"obs": "obs3", "action": "action3", "score": 3},
        {"obs": "obs4", "action": "action4", "score": 4, "token_usage": 12345},
        {"obs": "obs5", "action": "action5", "score": 5},
    ]

    # get should return the last 3 steps
    assert ht.get() == [
        {"obs": "obs3", "action": "action3", "score": 3},
        {"obs": "obs4", "action": "action4", "score": 4, "token_usage": 12345},
        {"obs": "obs5", "action": "action5", "score": 5},
    ]

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
    assert ht.json(3) == {
        "step_id": 3,
        "action": "action4",
        "obs": "obs4",
        "token_usage": 12345,
    }

    # json should return also the prompt-response pairs if include_prompt_response_pairs is True
    assert ht.json(2, include_prompt_response_pairs=True) == {
        "step_id": 2,
        "action": "action3",
        "obs": "obs3",
        "prompt_response_pairs": {
            "prompt_0": "prompt_3_1",
            "response_0": "response_3_1",
            "prompt_1": "prompt_3_2",
            "response_1": "response_3_2",
        },
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

    # should reset properly
    ht.reset()
    assert len(ht) == 0
    assert ht.get() == []
    assert ht.get_all() == []
    assert ht.score() == 0
    assert ht.prompt_response_pairs == [[]]

    # json should return an empty dict
    assert ht.json() == {}


def test_load_config():
    import atexit
    import tempfile
    from pathlib import Path

    import yaml

    # do the test in a tmp folder
    tempdir = tempfile.TemporaryDirectory(prefix="TestLoadConfig-")
    working_dir = Path(tempdir.name)
    config_file = working_dir / "config.yaml"
    atexit.register(tempdir.cleanup)  # Make sure to cleanup that folder once done.

    config_contents = {}
    config_contents["zero_shot"] = {
        "random_seed": 42,
        "max_steps": 100,
        "llm_name": "gpt2",
        "llm_temperature": [0.5],
    }
    config_contents["cot"] = {
        "random_seed": 43,
        "max_steps": 50,
        "cot_style": "standard",
        "llm_name": "gpt20",
        "llm_temperature": [0.3, 0.5],
    }

    # write the config file into yaml
    with open(config_file, "w") as f:
        yaml.dump(config_contents, f)

    # now test
    with patch(
        "sys.argv",
        [
            "config_file",
            str(config_file),
            "--agent",
            "zero_shot",
            "-p",
            "zero_shot.random_seed=123",
            "cot.llm_temperature=[0.8, 0.8]",
            "-v",
            "--debug",
        ],
    ):

        _config, _args = load_config()
    assert _args.agent == "zero_shot"
    assert "zero_shot" in _config.keys()
    assert "cot" in _config.keys()
    assert _config["zero_shot"]["random_seed"] == 123
    assert _config["zero_shot"]["max_steps"] == 100
    assert _config["zero_shot"]["llm_name"] == "gpt2"
    assert _config["zero_shot"]["llm_temperature"] == [0.5]
    assert _config["cot"]["random_seed"] == 43
    assert _config["cot"]["max_steps"] == 50
    assert _config["cot"]["cot_style"] == "standard"
    assert _config["cot"]["llm_name"] == "gpt20"
    assert _config["cot"]["llm_temperature"] == [0.8, 0.8]
    assert _args.debug is True
    assert _args.verbose is True


def test_build_history_prompt():
    import json

    from froggy.utils import unescape

    # test with empty history
    ht = HistoryTracker(history_steps=3)
    # use_conversational_prompt is False
    messages = _build_history_prompt(ht, use_conversational_prompt=False)
    expected = [
        {"role": "user", "content": "No history of command and terminal outputs."}
    ]
    assert messages == expected
    # use_conversational_prompt is True
    messages = _build_history_prompt(ht, use_conversational_prompt=True)
    expected = [
        {"role": "user", "content": "No history of command and terminal outputs."}
    ]
    assert messages == expected

    # test with non-empty history
    ht = HistoryTracker(history_steps=3)
    # push some steps
    ht.step({"obs": "obs1", "action": None, "score": 1, "rewrite_counter": 0})
    ht.step({"obs": "obs2", "action": "action2", "score": 2, "rewrite_counter": 0})
    ht.step({"obs": "obs3", "action": "action3", "score": 3, "rewrite_counter": 0})
    ht.step(
        {
            "obs": "obs4",
            "action": "action4",
            "score": 4,
            "token_usage": 12345,
            "rewrite_counter": 1,
        }
    )
    ht.step({"obs": "obs5", "action": "action5", "score": 5, "rewrite_counter": 1})

    # use_conversational_prompt is False
    # reset_prompt_history_after_rewrite is False
    messages = _build_history_prompt(
        ht, use_conversational_prompt=False, reset_prompt_history_after_rewrite=False
    )
    expected = [f"History of command and terminal outputs (the last 3 steps):"]
    history_messages = [
        {"step": 2, "command": "action3", "stdout": "obs3"},
        {"step": 3, "command": "action4", "stdout": "obs4"},
        {"step": 4, "command": "action5", "stdout": "obs5"},
    ]
    expected += ["\n" + unescape(json.dumps(history_messages, indent=4)) + "\n"]
    expected = [{"role": "user", "content": "\n".join(expected)}]
    assert messages == expected

    # reset_prompt_history_after_rewrite is True
    messages = _build_history_prompt(
        ht, use_conversational_prompt=False, reset_prompt_history_after_rewrite=True
    )
    expected = [f"History of command and terminal outputs (the last 2 steps):"]
    history_messages = [
        {"step": 3, "command": "action4", "stdout": "obs4"},
        {"step": 4, "command": "action5", "stdout": "obs5"},
    ]
    expected += ["\n" + unescape(json.dumps(history_messages, indent=4)) + "\n"]
    expected = [{"role": "user", "content": "\n".join(expected)}]
    assert messages == expected

    # use_conversational_prompt is True
    # reset_prompt_history_after_rewrite is False
    messages = _build_history_prompt(
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
    messages = _build_history_prompt(
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
