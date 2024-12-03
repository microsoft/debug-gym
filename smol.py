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
import copy

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
device = "cuda" if torch.cuda.is_available() else "cpu"


num_epochs = 1
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


def clean_model_name_for_tokenizer(model_name):
    if model_name.startswith("/root"):
        model_name = "-".join("/".join(model_name.split("/")[-2:]).split("-")[:-1])
    return model_name

class PPOLLM:
    def __init__(self, model_name, verbose=False):

        self.verbose = verbose

        config = PPOConfig(
            learning_rate=1.41e-5,
            batch_size=batch_size,
            mini_batch_size=1,
            seed=42
            # optimize_cuda_cache = True
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        # LoRA, commented out for now
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        # model = get_peft_model(model, peft_config)
        # model.print_trainable_parameters()
        # model.save_pretrained("/root/lora_model/")
        # model = PeftModel.from_pretrained(model, "/root/lora_model/", is_trainable=False, torch_dtype=torch.bfloat16)
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, is_trainable=True
        )

        model_name = clean_model_name_for_tokenizer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.token_counter = TokenCounter(model_name)

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
            "max_new_tokens": 500,
            "temperature": 0.5,
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
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

def write_to_file(file_path, data):
    with open(file_path, 'w') as file:
        for d in data:
            file.write(f"{d}\n")

def read_from_file(file_path):
    with open(file_path, 'r') as file:
        problems = file.read().splitlines()
    return problems

def run_with_ppo(agent, num_epochs, batch_size, problem_list):

    problem_iter = iter(problem_list)
    losses = []
    for epoch in range(num_epochs):
        while True:
            try:
                query_tensors, response_tensors, rewards = [], [], []
                while len(query_tensors) < batch_size:
                    
                    problem = next(problem_iter)
                    done = agent.run(task_name=problem)
                    query_tensors, response_tensors = agent.llm.get_tensors()

                    rewards += [torch.tensor([done]).float()] * (len(query_tensors) - len(rewards))

                    # optionally apply patch
                    if config["save_patch"]:
                        agent.save_patch(task_name=problem)
                    # save log
                    agent.log(task_name=problem)

                torch.cuda.empty_cache()
                gc.collect()

                stats = agent.llm.ppo_trainer.step(query_tensors[:batch_size], response_tensors[:batch_size], rewards[:batch_size])
                losses.append((stats['ppo/loss/value'], stats['ppo/loss/total']))
                print(stats['ppo/loss/value'], stats['ppo/loss/total'])
                agent.llm.clear_tensors()
                torch.cuda.empty_cache()
                gc.collect()

            except StopIteration:
                break
    return losses

def run_without_ppo(agent, problem_list):

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

    # /root/models/Qwen/Qwen2.5-Coder-0.5B-Instruct-PPO

    models = {
        "q0.5": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "q1.5": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "q3": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "q7": "Qwen/Qwen2.5-Coder-7B-Instruct"
    }


    # TODO: Fix/robustify argument parsing when time permits 
    if len(sys.argv) > 2:

        # Get command line arguments 
        input, model_name = sys.argv[1], sys.argv[2]
        try:
            model_name = models[model_name]
        except KeyError:
            print("Model not found")

        # Setup agent and environment
        env = create_env(None, config)
        agent = AgentZeroShot_NoPDB(config, env, verbose=False)

        # Specify HuggingFace model and hijack agent
        agent.llm = PPOLLM(model_name, verbose=False)

        # Choose whether to use full dataset or 'easy' dataset based on flag
        if "-easy" in sys.argv:
            problem_list = read_from_file("/root/Froggy/easy_problems.txt")
        else:
            problem_list = list(env.dataset.keys())
        
        # Run with PPO training or without
        if input == "ppo":
            losses = run_with_ppo(agent, num_epochs, batch_size, problem_list)
            agent.llm.ppo_trainer._save_pretrained(f"/root/models/{model_name}-PPO")
            write_to_file(f"/root/models/{model_name}-PPO/losses.txt", losses)
        elif input == "baseline":
            run_without_ppo(agent, problem_list)
        else:
            print("Unrecognized input.")
    else:
        print("Incomplete inputs provided.")

if __name__ == "__main__":
    main()