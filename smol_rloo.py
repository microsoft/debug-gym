import numpy as np
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
)
from trl.core import LengthSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from froggy.agents.zero_shot import AgentZeroShot, AgentZeroShot_NoPDB
from froggy.agents.llm_api import LLM, merge_messages, print_messages
from froggy.agents.utils import trim_prompt_messages
from froggy.envs import AiderBenchmarkEnv
from scripts.run import create_env
import torch
import tiktoken
import os
from os.path import join as pjoin
from termcolor import colored
import gc

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "output_path": "./root",
    "benchmark": "aider",
    "problems": "all",  # list of problems, e.g., ["wordy"], or "all"
    "env_kwargs": {
        "dir_tree_depth": 1,
        "run_on_rewrite": True,
        "auto_view_change": True,
        "run_timeout": 10,
    },
    "tools": ["patcher:whole"],
    "terminal": {
        "type": "local",  # "docker" or "local"
        "base_image": "python:3.12-slim",
        "setup_commands": ["pip install pytest"],
    },
    "persistent_breakpoints": True,  # in pdb tool
    # LLM configs
    "llm_name": "random",
    "llm_temperature": [0.5],  # list of values between 0.0 and 1.0
    # Agent configs
    "random_seed": 42,
    "max_steps": 2,
    "max_rewrite_steps": 10,
    "memory_size": 20,
    "use_conversational_prompt": True,
    "save_patch": True,
    "log_prompt_response_pairs": True,
    "reset_prompt_history_after_rewrite": True,
}


batch_size = 4
env = create_env(None, config)
agent = AgentZeroShot_NoPDB(config, env, verbose=False)


class TokenCounter:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        try:
            self.tokenize = tiktoken.encoding_for_model(model).encode
        except KeyError:
            try:
                # Try to load from transformers.
                self.tokenize = AutoTokenizer.from_pretrained(model).tokenize
            except OSError:
                msg = (
                    f"Tokenizer not found for model {model},"
                    " make sure you have access to the model"
                    " (e.g., HuggingFace API key is correctly set)."
                )
                raise ValueError(msg)

    def __call__(self, *, messages=None, text=None):
        nb_tokens = 0
        if messages is not None:
            nb_tokens += sum(len(self.tokenize(msg["content"])) for msg in messages)

        if text is not None:
            nb_tokens += len(self.tokenize(text))

        return nb_tokens


@torch.no_grad()
def generate_episodes(agent, env):
    import numpy as np
    from trl.trainer.utils import pad

    problems = list(env.dataset.keys())

    rewards = []
    logits = []
    query_tensors = []
    response_tensors = []

    while True:
        problem = np.random.choice(problems)
        done = agent.run(task_name=problem)

        query_tensors_, response_tensors_, logits_ = agent.llm.get_tensors()
        query_tensors += query_tensors_
        response_tensors += response_tensors_
        logits += logits_
        rewards += [torch.tensor([done]).float()] * (len(query_tensors) - len(rewards))

        if len(query_tensors) >= batch_size:
            break

    query_tensors = query_tensors[:batch_size]
    response_tensors = response_tensors[:batch_size]
    logits = logits[:batch_size]
    rewards = torch.cat(rewards[:batch_size], 0).to(device)
    breakpoint()

    qt = pad(
        query_tensors,
        agent.llm.tokenizer.pad_token_id,
        padding_side="left",
    )
    rt = pad(
        response_tensors,
        agent.llm.tokenizer.pad_token_id,
        padding_side="right",
    )
    logits = pad(logits, 0, padding_side="right")

    context_length = qt.shape[1]
    qrt = torch.cat((qt, rt), dim=1)
    all_logprob = torch.nn.functional.log_softmax(logits, dim=-1)
    logprob = torch.gather(all_logprob, 2, rt.unsqueeze(-1)).squeeze(-1)

    del logits, all_logprob
    torch.cuda.empty_cache()

    attention_mask = qrt != agent.llm.tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(qrt, ~attention_mask, 0)
    ref_output = agent.llm.ref_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
    )
    ref_logits = ref_output.logits[:, context_length - 1 : -1]
    ref_logits /= config['llm_temperature'][0] + 1e-7
    ref_all_logprob = torch.nn.functional.log_softmax(ref_logits, dim=-1)
    ref_logprob = torch.gather(ref_all_logprob, 2, rt.unsqueeze(-1)).squeeze(-1)
    del ref_output, ref_logits, ref_all_logprob
    torch.cuda.empty_cache()

    kl = logprob - ref_logprob
    non_score_reward = (-0.1 * kl).sum(1)
    rlhf_reward = rewards + non_score_reward

    # vectorized RLOO advantages implementation
    advantages = rlhf_reward.flatten()
    torch.cuda.empty_cache()
    agent.llm.clear_tensors()
    return advantages, logprob, qt, rt, qrt


class RLOOLLM:
    def __init__(self, model_name, verbose=False):

        self.verbose = verbose
        self.token_counter = TokenCounter(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.query_tensors = []
        self.response_tensors = []
        self.logits = []

    def tokenize_messages(self, messages):
        messages_tokens = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        return torch.tensor(messages_tokens)

    def get_tensors(self):
        return self.query_tensors, self.response_tensors, self.logits

    def clear_tensors(self):
        self.query_tensors = []
        self.response_tensors = []

    def __call__(self, messages, *args, **kwargs):
        # Merge consecutive messages with same role.
        messages = merge_messages(messages)
        messages = trim_prompt_messages(messages, 32768, self.token_counter)

        if self.verbose:
            print_messages(messages)

        tokenized_messages = self.tokenize_messages(messages).unsqueeze(0).to(device)

        generation_kwargs = {
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 500,
            "temperature": 0.5,
            "top_p": 1.0,
        }

        context_length = tokenized_messages.shape[1]
        outputs = self.model.generate(
            tokenized_messages,
            **generation_kwargs,
            return_dict_in_generate=True,
            output_scores=True,
        )
        logits = torch.stack(outputs.scores, 1).squeeze()
        response_tensor = outputs.sequences[:, context_length:].squeeze()

        response = self.tokenizer.decode(response_tensor, skip_special_tokens=True)

        self.logits.append(logits)
        self.query_tensors.append(tokenized_messages.squeeze())
        self.response_tensors.append(response_tensor)
        torch.cuda.empty_cache()
        gc.collect()

        token_usage = {
            "prompt": self.token_counter(messages=messages),
            "response": self.token_counter(text=response),
        }
        return response, token_usage


model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
agent.llm = RLOOLLM(model_name, verbose=False)
optimizer = torch.optim.Adam(agent.llm.model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

# Getting the problem list
problem_list = env.dataset.keys()


for batch in range(100):
    advantages, logprobs, qt, rt, qrt = generate_episodes(agent, env)
    context_length = qt.shape[1]

    for ppo_epoch_idx in range(4):
        b_inds = np.random.permutation(batch_size)

        for step in range(batch_size // 2):
            mini_batch_inds = b_inds[step : step + 2]
            mb_advantage = advantages[mini_batch_inds]
            mb_responses = rt[mini_batch_inds]
            mb_query_responses = qrt[mini_batch_inds]
            mb_logprobs = logprobs[mini_batch_inds]

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                attention_mask = mb_query_responses != agent.llm.tokenizer.pad_token_id
                position_ids = attention_mask.cumsum(1) - attention_mask.long()
                input_ids = torch.masked_fill(mb_query_responses, ~attention_mask, 0)
                output = agent.llm.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                )

                logits = output.logits[:, context_length - 1 : -1]
                logits /= config['llm_temperature'][0] + 1e-7
                new_all_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                new_logprobs = torch.gather(
                    new_all_logprobs, 2, mb_responses.unsqueeze(-1)
                ).squeeze(-1)
                # padding?
                new_ratio = (new_logprobs - mb_logprobs).exp()
                new_logprobs = new_logprobs.sum(1)
                mb_logprobs = mb_logprobs.sum(1)
                logprobs_diff = new_logprobs - mb_logprobs
                ratio = torch.exp(logprobs_diff)
                pg_losses = -mb_advantage * ratio
                pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0, 1.0)
                pg_loss_max = torch.max(pg_losses, pg_losses2)
                pg_loss = pg_loss_max.mean()
                loss = pg_loss / batch_size
                loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(f"Batch {batch}, PPO epoch {ppo_epoch_idx}, loss: {loss.item()}, rewards: {advantages.mean().item()}")
