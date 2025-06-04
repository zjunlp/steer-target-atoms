# package import
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from model_wrapper import (
    QwenWrapper,
)
import random
import numpy as np
import torch

random.seed(42)  # 设置 Python 随机数种子
np.random.seed(42)  # 设置 NumPy 随机数种子
torch.manual_seed(42)  # 设置 PyTorch 随机数种子
torch.cuda.manual_seed_all(42)

def trim_trailing_duplicates(output_tensor: torch.Tensor) -> torch.Tensor:
    if output_tensor.numel() == 0:
        return output_tensor
    # 如果是1维，就保持原来的逻辑
    if output_tensor.dim() == 1:
        last_token = output_tensor[-1]
        index = output_tensor.size(0) - 1
        while index >= 0 and output_tensor[index] == last_token:
            index -= 1
        return output_tensor[:index+1]
    # 如果是二维，则沿着最后一个维度去除尾部重复，同时保持二维形式
    elif output_tensor.dim() == 2:
        last_token = output_tensor[0, -1]
        index = output_tensor.size(1) - 1
        while index >= 0 and output_tensor[0, index] == last_token:
            index -= 1
        return output_tensor[:, :index+1]
    else:
        raise ValueError("仅支持1维和2维的tensor")
    
model_name_or_path="/data2/xzwnlp/model/DeepSeek-R1-Distill-Qwen-7B"
device = "cuda:5"
model = QwenWrapper(model_name_or_path=model_name_or_path, device = device)
tokenizer = model.tokenizer

layer=17

results_all = []

from datasets import Dataset, load_dataset
dataset = load_dataset("json", data_files="./data/safety/gsm/test.jsonl", split="train")
all_ques = dataset[:1]["question"]

for i, ques in enumerate(all_ques):
    print("Now id is ",i)
    results={}
    results["dataset"] = "gsm8k"
    results["question"] = ques
    results["id"] = i
    for multiplier in range(-3,-2):
        caa_vector_path = os.path.join(
            "./data/r1/overthink_qwen/caa_vector/deepseek-qwen-7b_r1", f"{layer}.pt"
        )
        caa_vector = torch.load(caa_vector_path).to(device)
        # print("Caa vector path: ",caa_vector_path)
        # print("Caa vector: ",caa_vector)
        # print("caa vector: ",torch.norm(caa_vector, p=2))
        model.reset_all()
        model.set_add_activations(
            layer, multiplier * caa_vector
        )
        ques = "Answer the following question.\n\n" + "Question: " + ques.strip() + "\n\nPlease wrap the final answer in $\\boxed{{}}$ tag."
        chat = [
            { "role": "user", "content": ques},
        ]
        prompt_tokens = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        # print(prompt_tokens)
        # print(f"{tokenizer.decode(prompt_tokens[0])}")

        prompt_tokens = prompt_tokens.to(device)  
        outputs={}
        token_lens={}

        for i in range(1):
            output = model.model.generate(prompt_tokens, temperature=0.1, max_length=131072)
            print(tokenizer.batch_decode(output))
            output = output[:,prompt_tokens.shape[-1]:]
            # print(output.shape)
            token_len_b = output.shape[-1]

            output = trim_trailing_duplicates(output)
            # print(output.shape)
            token_len_a = output.shape[-1]
            output = tokenizer.batch_decode(output)
            # print(output[0])
            outputs[i] = output[0]
            token_lens[i] = (token_len_b, token_len_a)

        results[multiplier] = {"outputs": outputs, "token_lens": token_lens}
    results_all.append(results)
import json
with open('./data/r1/overthink_qwen/results/results_qwen_ques_gsm_top20_add_gsm_template_max_len.json', 'w') as json_file:
    json.dump(results_all, json_file, indent=4)