from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
from trl.core import LengthSampler
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from froggy.agents.zero_shot import AgentZeroShot
from froggy.agents.llm_api import LLM, merge_messages, print_messages
from froggy.agents.utils import trim_prompt_messages
from froggy.envs import AiderBenchmarkEnv
import torch
import tiktoken
import os
from os.path import join as pjoin
from termcolor import colored
import gc

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Aider configurations and env
env = AiderBenchmarkEnv()
config = {
    "output_path": "/root",
    "benchmark": "aider",
    "problems": "all",  # list of problems, e.g., ["wordy"], or "all"
    "env_kwargs": {
        "dir_tree_depth": 1,
        "run_on_rewrite": True,
        "auto_view_change": True,
        "run_timeout": 10,
    },
    "tools": ["pdb", "view", "patcher:substitution"],
    "persistent_breakpoints": True,  # in pdb tool

    # LLM configs
    "llm_name": "random",
    "llm_temperature": [0.0],  # list of values between 0.0 and 1.0

    # Agent configs
    "random_seed": 42,
    "max_steps": 25,
    "max_rewrite_steps": 10,
    "memory_size": 20,
    "use_conversational_prompt": True,
    "save_patch": True,
    "log_prompt_response_pairs": True,
    "reset_prompt_history_after_rewrite": True
}


agent = AgentZeroShot(config, env)


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


class PPOLLM():
  def __init__(self, model_name, verbose=False):

    self.verbose = verbose
    self.token_counter = TokenCounter(model_name)

    config = PPOConfig(
        learning_rate = 1.41e-5,
        batch_size = 25,
        mini_batch_size = 1,
        # optimize_cuda_cache = True
    )

    peft_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM, 
      inference_mode=False, 
      r=8, 
      lora_alpha=32, 
      lora_dropout=0.1
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.save_pretrained("/root/lora_model/") 
    model = PeftModel.from_pretrained(model, "/root/lora_model/", is_trainable=False, torch_dtype=torch.bfloat16)
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model, torch_dtype=torch.bfloat16, is_trainable=True)

    # ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.ppo_trainer = PPOTrainer(
        config = config,
        model = ppo_model,
        tokenizer = self.tokenizer
    )

    self.query_tensors, self.response_tensors, self.rewards = [], [], []

  def tokenize_messages(self, messages):

    messages_tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        
    return torch.tensor(messages_tokens)
  
  def get_tensors(self):
     return self.query_tensors, self.response_tensors, self.rewards
     clear_tensors

  def clear_tensors(self):
     self.query_tensors = []
     self.response_tensors = []
     self.rewards = []
      
  def __call__(self, messages, *args, **kwargs):

    # Merge consecutive messages with same role.
    messages = merge_messages(messages)
    messages = trim_prompt_messages(
        messages, 32768, self.token_counter
    )

    if self.verbose:
        print_messages(messages)
    
    tokenized_messages = self.tokenize_messages(messages)

    generation_kwargs = {
        "do_sample": True,
        "eos_token_id": self.tokenizer.eos_token_id,
        "max_new_tokens": 500
    }

    self.query_tensors.append(tokenized_messages)
    response_tensor = self.ppo_trainer.generate(tokenized_messages.to(device), pad_token_id=self.tokenizer.pad_token_id, **generation_kwargs).squeeze()
    response_tensor = response_tensor[tokenized_messages.shape[0]:]
    response = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
    self.response_tensors.append(response_tensor)
    self.rewards.append(torch.tensor([0]).float())
    print(response)

    tokenized_messages = tokenized_messages.detach().to('cpu')
    torch.cuda.empty_cache()
    gc.collect()

    token_usage = {
      "prompt": self.token_counter(messages=messages),
      "response": self.token_counter(text=response),
    }

    return response, token_usage

# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
agent.llm = PPOLLM(model_name, verbose=False)

# Getting the problem list 
problem_list = env.dataset.keys() 

for problem in problem_list:

  if os.path.exists(pjoin(agent._output_path, problem, "froggy.jsonl")):
    print(colored(f"Skipping {problem}, already done.", "yellow"))
    continue
  print(
    colored(
        f"Running agent {agent.name} on {config['benchmark']}.{problem}",
        "green",
    )
  )
  done = agent.run(task_name=problem)

  query_tensors, response_tensors, rewards = agent.llm.get_tensors()

  # for i in range(len(query_tensors)):
  #   print(query_tensors[i].shape, response_tensors[i].shape, "||||")

  if done:
    rewards = [torch.tensor([1]).float()]*len(rewards)

  print(done)
  
  torch.cuda.empty_cache()
  gc.collect()

  stats = agent.llm.ppo_trainer.step(query_tensors, response_tensors, rewards)

  torch.cuda.empty_cache()
  gc.collect()

  agent.llm.clear_tensors()

  # optionally apply patch
  if config["save_patch"]:
    agent.save_patch(task_name=problem)
  # save log
  agent.log(task_name=problem)