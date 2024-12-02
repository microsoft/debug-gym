from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead
)
from trl.core import LengthSampler
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from froggy.agents.zero_shot import AgentZeroShot, AgentZeroShot_NoPDB
from froggy.agents.llm_api import LLM, merge_messages, print_messages
from froggy.agents.utils import trim_prompt_messages
from froggy.envs import AiderBenchmarkEnv
from scripts.run import create_env
import torch
import tiktoken
import sys
from os.path import join as pjoin
from termcolor import colored
import gc
import random

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
device = "cuda" if torch.cuda.is_available() else "cpu"


num_epochs = 10
batch_size = 10
config = {
    "output_path": "/root/outputs",
    "benchmark": "aider",
    "problems": "all",  # list of problems, e.g., ["wordy"], or "all"
    "env_kwargs": {
        "dir_tree_depth": 1,
        "run_on_rewrite": True,
        "auto_view_change": True,
        "run_timeout": 10,
        "entrypoint": "python -m pytest --maxfail=1 ."
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


class PPOLLM:
    def __init__(self, model_name, verbose=False):

        self.verbose = verbose
        self.token_counter = TokenCounter(model_name)

        config = PPOConfig(
            learning_rate=1.41e-5,
            batch_size=batch_size,
            mini_batch_size=1,
            # optimize_cuda_cache = True
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        # model = get_peft_model(model, peft_config)
        # model.print_trainable_parameters()
        # model.save_pretrained("/root/lora_model/")
        # model = PeftModel.from_pretrained(model, "/root/lora_model/", is_trainable=False, torch_dtype=torch.bfloat16)
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, is_trainable=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.ppo_trainer = PPOTrainer(
            config=config, model=ppo_model, tokenizer=self.tokenizer
        )

        self.query_tensors, self.response_tensors, self.rewards = [], [], []

    def tokenize_messages(self, messages):

        messages_tokens = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )

        return torch.tensor(messages_tokens)

    def get_tensors(self):
        return self.query_tensors, self.response_tensors

    def clear_tensors(self):
        self.query_tensors = []
        self.response_tensors = []
        self.rewards = []

    def __call__(self, messages, *args, **kwargs):

        # Merge consecutive messages with same role.
        messages = merge_messages(messages)
        messages = trim_prompt_messages(messages, 32768, self.token_counter)

        if self.verbose:
            print_messages(messages)

        tokenized_messages = self.tokenize_messages(messages)

        generation_kwargs = {
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 500,
            "temperature": 0.5,
        }

        self.query_tensors.append(tokenized_messages)
        response_tensor = self.ppo_trainer.generate(
            tokenized_messages.to(device),
            pad_token_id=self.tokenizer.pad_token_id,
            **generation_kwargs,
        ).squeeze()
        response_tensor = response_tensor[tokenized_messages.shape[0] :]
        response = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
        self.response_tensors.append(response_tensor)
        # print(response)

        tokenized_messages = tokenized_messages.detach().to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        token_usage = {
            "prompt": self.token_counter(messages=messages),
            "response": self.token_counter(text=response),
        }

        return response, token_usage

def run_with_ppo(agent, env, num_epochs, batch_size):

    # Getting the problem list
    problem_list = list(env.dataset.keys())
    for epoch in range(num_epochs):
        query_tensors, response_tensors, rewards = [], [], []
        while len(query_tensors) < batch_size:
            idx = random.randint(0, len(problem_list))
            problem = problem_list[idx]
            done = agent.run(task_name=problem)
            query_tensors, response_tensors = agent.llm.get_tensors()

            rewards += [torch.tensor([done]).float()] * (len(query_tensors) - len(rewards))

            torch.cuda.empty_cache()
            gc.collect()

        stats = agent.llm.ppo_trainer.step(query_tensors[:batch_size], response_tensors[:batch_size], rewards[:batch_size])
        agent.llm.clear_tensors()
        torch.cuda.empty_cache()
        gc.collect()

        # optionally apply patch
        if config["save_patch"]:
            agent.save_patch(task_name=problem)
        # save log
        agent.log(task_name=problem)

def run_without_ppo(agent, env):

    problem_list = list(env.dataset.keys())
    for problem in problem_list:

        print(
            colored(
                f"Running agent {agent.name} on {config['benchmark']}.{problem}",
                "green",
            )
        )
        done = agent.run(task_name=problem)

        # optionally apply patch
        if config["save_patch"]:
            agent.save_patch(task_name=problem)
        # save log
        agent.log(task_name=problem)

def main():

    # Setup agent and environment
    env = create_env(None, config)
    agent = AgentZeroShot_NoPDB(config, env, verbose=False)

    # Specify HuggingFace model and hijack agent
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    agent.llm = PPOLLM(model_name, verbose=False)

    if len(sys.argv) > 1:
        input = sys.argv[1]
        if input == "ppo":
            run_with_ppo(agent, env, num_epochs, batch_size)
        elif input == "baseline":
            run_without_ppo(agent, env)
        else:
            print("Unrecognized input.")

    else:
        print("No input provided.")

if __name__ == "__main__":
    main()