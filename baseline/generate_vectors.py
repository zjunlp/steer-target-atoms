import sys

sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from model_wrapper import (
    QwenWrapper,
    LlamaWrapper,
    GemmaWrapper,
)
import argparse
from typing import List
from utils.input_format import llama3_chat_input_format_train
from dataloader import (
    GenerationDataset, 
    YNDataset, 
    PersonalityEditDataset,
    SafetyDataset
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


def prepare_input(tokenizer, prompts, device="cuda"):
    input_tokens = tokenizer.tokenize(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)

    return input_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/20t/msy/shae/data/generation",
    )
    parser.add_argument("--data_name", type=str, default="power-seeking")
    parser.add_argument("--mode", type=str, default="toxic")
    parser.add_argument("--data_type", type=str, default="generation")
    parser.add_argument("--model_name", type=str, default="gemma-2-9b")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="gemma-2-9b")
    parser.add_argument("--AB", action="store_true", default=False)

    args = parser.parse_args()
    print(args)
    if "llama" in args.model_name.lower():
        model = (
            LlamaWrapper(args.model_name_or_path)
        )
    elif "gemma" in args.model_name.lower():
        model = (
            GemmaWrapper(args.model_name_or_path)
        )
    elif "qwen" in args.model_name.lower():
        model = (
            QwenWrapper(args.model_name_or_path)
        )
    else:
        raise NotImplementedError(f"Model {args.model_name} not supported")
    tokenizer = model.tokenizer
    pos_activations = dict([(layer, []) for layer in args.layers])
    neg_activations = dict([(layer, []) for layer in args.layers])

    if args.data_type == "generation":
        dataset = GenerationDataset()
    elif args.data_type == "yn":
        dataset = YNDataset()
    elif args.data_type == "personalityedit":
        dataset = PersonalityEditDataset()
    elif args.data_type == "safety":
        dataset = GenerationDataset()
    else:
        raise ValueError("Invalid data type")

    vector_dataset = dataset.get_data_for_caa(
        data_path=args.data_path,
        data_name=args.data_name,
        split="train",
    )
    device = model.device

    pos_tokens_list, neg_tokens_list = [], []

    for i in range(len(vector_dataset)):
        if args.AB == True and args.model_name=="gemma-2-9b-it":
            ques = "<start_of_turn>user\nQurstion: " + vector_dataset[i]["question"] + "<end_of_turn>\n<start_of_turn>model"
            chosen = "\nAnswer:" + vector_dataset[i]["chosen"]
            rejected = "\nAnswer:" + vector_dataset[i]["rejected"]
        elif args.AB == True and args.model_name=="gemma-2-9b":
            ques = "Question: "+ vector_dataset[i]["question"]
            chosen = "\nAnswer:" + vector_dataset[i]["chosen"]
            rejected = "\nAnswer:" + vector_dataset[i]["rejected"]
        else:
            ques = vector_dataset[i]["question"]
            chosen = vector_dataset[i]["chosen"]
            rejected = vector_dataset[i]["rejected"]

        if ques and chosen and rejected:
            
            if args.model_name=="gemma-2-9b" or args.model_name=="llama-3.1":
                if args.system_prompt != "":
                    if ques is not None:
                        ques = args.system_prompt + " " + ques
                    else:
                        ques = args.system_prompt
            elif args.model_name=="gemma-2-9b-it" and args.AB == False:
                if args.system_prompt != "":
                    print('system_prompt for gemma-2-9b-it is not done')
                    raise NotImplementedError
                else:
                    ques = f"<start_of_turn>user\n{ques}<end_of_turn>\n<start_of_turn>model\n"
                    # print("ques:", ques)
            else:
                raise NotImplementedError


            ques_tokens = tokenizer.encode(ques, return_tensors="pt")
            pos_tokens = tokenizer.encode(ques + chosen, return_tensors="pt")
            neg_tokens = tokenizer.encode(ques + rejected, return_tensors="pt")
            pos_tokens_list.append(
                {
                    "pos_tokens": pos_tokens.to(device),
                    "ques_tokens_len": ques_tokens.shape[1],
                    "pos_answer_len": pos_tokens.shape[1] - ques_tokens.shape[1],
                }
            )
            neg_tokens_list.append(
                {
                    "neg_tokens": neg_tokens.to(device),
                    "ques_tokens_len": ques_tokens.shape[1],
                    "neg_answer_len": neg_tokens.shape[1] - ques_tokens.shape[1],
                }
            )

    for p_tokens_dict, n_tokens_dict in tqdm(
        zip(pos_tokens_list, neg_tokens_list),
        total=len(pos_tokens_list),
        desc="Processing prompts",
    ):
        p_tokens = p_tokens_dict["pos_tokens"]
        n_tokens = n_tokens_dict["neg_tokens"]
        ques_tokens_len = p_tokens_dict["ques_tokens_len"]
        model.reset_all()
        model.get_logits(p_tokens)

        for layer in args.layers:
            p_activations = model.get_last_activations(layer)
            # mean the activation over all answer tokens
            if args.AB == True:
                p_activations = p_activations[0, -2, :].detach().cpu()
            else:
                p_activations = p_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
            pos_activations[layer].append(p_activations)

        model.reset_all()
        model.get_logits(n_tokens)

        for layer in args.layers:
            n_activations = model.get_last_activations(layer)
            if args.AB == True:
                n_activations = n_activations[0, -2, :].detach().cpu()
            else:
                n_activations = n_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
            neg_activations[layer].append(n_activations)

    if args.model_name == "gemma-2-9b-it":
        output_dir = os.path.join(
            args.data_path, args.data_name, "caa_vector_it", f"{args.model_name}_{args.mode}"
        )
    elif args.model_name == "gemma-2-9b" or args.model_name=="llama-3.1":
        output_dir = os.path.join(
            args.data_path, args.data_name, "caa_vector_pt", f"{args.model_name}_{args.mode}"
        )
    else:
        output_dir = os.path.join(
            args.data_path, args.data_name, "caa_vector", f"{args.model_name}_{args.mode}"
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for layer in args.layers:
        all_pos_layer = torch.stack(pos_activations[layer])
        all_neg_layer = torch.stack(neg_activations[layer])
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)

        torch.save(
            vec,
            os.path.join(output_dir, f"{layer}.pt"),
        )
