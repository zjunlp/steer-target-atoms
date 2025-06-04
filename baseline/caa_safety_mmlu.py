import sys
import gc

sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

import json
from model_wrapper import (
    LlamaWrapper,
    GemmaWrapper,
)
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from utils.helpers import get_a_b_probs
from utils.tokenize import E_INST
import torch
from dataloader import GenerationDataset, YNDataset, MMLUDataset
from utils.infer_utils import clean_preds, batch_infer
import pdb

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--mode", type=str, default="toxic")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--model_name", type=str, default="llama-3.1")
    parser.add_argument("--model_name_or_path", type=str, default="llama-3.1")
    parser.add_argument("--data_path", type=str, default="/data2/xzwnlp/SaeEdit/git/shae/data/safety")
    parser.add_argument("--data_name", type=str, default="toxic_DINM")
    parser.add_argument("--eval_data_name", type=str, default="realtoxicity")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--caa_vector_root", type=str, required=True)
    parser.add_argument("--AB", action="store_true", default=False)
    parser.add_argument("--qa", action="store_true", default=False)
    
    args = parser.parse_args()
    print(args)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
    else:
        print(f"Using {torch.cuda.device_count()} GPUs!")

    model = LlamaWrapper(args.model_name_or_path) if args.model_name == "llama-3.1" else GemmaWrapper(args.model_name_or_path)
    tokenizer = model.tokenizer

    mmlu = MMLUDataset()

    mmlu_test = mmlu.get_test_data()
    # mmlu_test = mmlu_test.select(range(1))


    device = model.device
  

    prompt_tokens_list = []
    
    vector_root = os.path.join(args.caa_vector_root)

    directory = os.path.dirname(args.output_file)  


    # pdb.set_trace()
    for i in tqdm(range(len(mmlu_test))):
        ques = mmlu_test[i]["prompt"]
        if args.model_name=="gemma-2-9b" or args.model_name=="llama-3.1":
            if args.system_prompt != "":
                if ques is not None:
                    ques = args.system_prompt + " " + ques
                else:
                    ques = args.system_prompt
            # prompt_tokens_list.append(tokenizer.encode(ques, return_tensors="pt").to(device))
            prompt_tokens_list.append(ques)
        elif args.model_name=="gemma-2-9b-it" and args.AB==True:
            if args.system_prompt != "":
                print('system_prompt for gemma-2-9b-it is not done')
                raise NotImplementedError
            else:
                ques = f"<start_of_turn>user\nQuestion: {ques}<end_of_turn>\n<start_of_turn>model\nAnswer:"
                prompt_tokens_list.append(ques)
        elif args.model_name=="gemma-2-9b-it" and args.AB==False:
            if args.system_prompt != "":
                print('system_prompt for gemma-2-9b-it is not done')
                raise NotImplementedError
            elif args.qa==True:
                ques = f"<start_of_turn>user\nQuestion: {ques}<end_of_turn>\n<start_of_turn>model\nAnswer:"
                # print("ques:",ques)
                prompt_tokens_list.append(ques)
            else:
                ques = f"<start_of_turn>user\n{ques}<end_of_turn>\n<start_of_turn>model\n"
                # print("ques:", ques)
                prompt_tokens_list.append(ques)
        else:
            raise NotImplementedError

    del mmlu_test
    # torch.cuda.empty_cache()


    if not os.path.exists(directory):  
        os.makedirs(directory)  
    for layer in args.layers:
        print(f"Layer {layer}")
        vector_path = os.path.join(
            vector_root, f"{layer}.pt"
        )

        steering_vector = torch.load(vector_path).to(device)
        print("Steering vector path: ",vector_path)
        print("Steering vector: ",steering_vector)
        for multiplier in args.multipliers:
            
            preds = []
            model.reset_all()
            print(f"Multiplier {multiplier}")

            model.set_add_activations(
                layer, multiplier * steering_vector
            )
            

            mmlu_preds = batch_infer(
                model.model,
                tokenizer,
                prompt_tokens_list,
                batch_size=1,
                max_new_tokens=1,
                return_score=True,
            )

            mmlu_acc = mmlu.get_accuracy(mmlu_preds, tokenizer=tokenizer)

            results = []

            results.append(
                {
                    "mmlu_acc": mmlu_acc,
                }
            )

            output_dir = os.path.dirname(args.output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = args.output_file.replace(".result.json", f"_multiplier{multiplier}.result.json")
            json.dump(
                results, open(output_file, "w"), indent=4, ensure_ascii=False
            )
