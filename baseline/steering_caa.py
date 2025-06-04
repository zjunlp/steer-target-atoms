import sys
import pdb

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
from dataloader import GenerationDataset, YNDataset
import pdb
from transformers import StoppingCriteria

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)
    
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--mode", type=str, default="toxic")
    parser.add_argument("--model_name", type=str, default="llama-3.1")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--trim", type=float, default=-1)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--system_prompt_back", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="llama-3.1")
    parser.add_argument("--data_name", type=str, default="power-seeking")
    parser.add_argument("--eval_data_name", type=str, default="power-seeking")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="/root/paddlejob/workspace/msy/shae/data")
    parser.add_argument("--AB", action="store_true", default=False)
    parser.add_argument("--qa", action="store_true", default=False)
    
    args = parser.parse_args()
    print(args)

    model = LlamaWrapper(args.model_name_or_path) if args.model_name == "llama-3.1" else GemmaWrapper(args.model_name_or_path)
    tokenizer = model.tokenizer

    dataset = GenerationDataset()
    if "toxigen" in args.eval_data_name:
        get_data=dataset.get_data_for_toxigen
    elif "realtoxicity" in args.eval_data_name:
        get_data=dataset.get_data_for_realtoxicity
    elif "gsm" in args.eval_data_name:
        get_data=dataset.get_data_for_gsm
    elif "prefilling" in args.eval_data_name:
        get_data=dataset.get_data_for_prefilling
    elif "exaggerated-safety" in args.eval_data_name:
        get_data=dataset.get_data_for_exaggerated_safety
    elif "prefilling" in args.eval_data_name:
        get_data=dataset.get_data_for_prefilling
    elif "nqopen" in args.eval_data_name:
        args.qa = True
        get_data=dataset.get_data_for_nqopen
    else:
        get_data=dataset.get_data_for_caa
        
    if "gsm" in args.eval_data_name and args.model_name == "gemma-2-9b-it":
            vector_dataset = get_data(
                data_path=args.data_path,
                data_name=args.eval_data_name,
                split="test",
                n_shots=0
            )
    else:
        vector_dataset = get_data(
            data_path=args.data_path,
            data_name=args.eval_data_name,
            split="test",
        )

    device = model.device

    prompt_tokens_list = []
    questions = vector_dataset["question"]
    if "toxigen" in args.eval_data_name:
        label = vector_dataset["label"]
    elif "gsm" in args.eval_data_name or "nqopen" in args.eval_data_name:
        answers = vector_dataset["answer"]
    elif "exaggerated-safety" in args.eval_data_name:
        ids = vector_dataset["id"]
        tys = vector_dataset["type"]
    for i in range(len(vector_dataset)):
        ques = vector_dataset[i]["question"]
        if not ques: continue
        if args.AB == True and "gsm" not in args.eval_data_name and args.model_name!="gemma-2-9b-it":
            ques = "Question: "+ ques + "\nAnswer:"
        if args.model_name=="gemma-2-9b" or args.model_name=="llama-3.1":
            if args.system_prompt != "":
                if ques is not None:
                    ques = args.system_prompt + " " + ques
                else:
                    ques = args.system_prompt
            prompt_tokens_list.append(tokenizer.encode(ques, return_tensors="pt").to(device))
        elif args.model_name=="gemma-2-9b-it" and args.AB==True:
            if args.system_prompt != "":
                print('system_prompt for gemma-2-9b-it is not done')
                raise NotImplementedError
            else:
                ques = f"<start_of_turn>user\nQuestion: {ques}<end_of_turn>\n<start_of_turn>model\nAnswer:"
                prompt_tokens_list.append(tokenizer.encode(ques, return_tensors="pt").to(device))
        elif args.model_name=="gemma-2-9b-it" and args.AB==False:
            if args.system_prompt != "":
                if ques is not None:
                    ques = args.system_prompt + " " + ques
                else:
                    ques = args.system_prompt
            if args.system_prompt_back != "":
                if ques is not None:
                    ques = ques + " " + args.system_prompt_back
                else:
                    ques = args.system_prompt_back
            
            if args.qa==True:
                ques = f"<start_of_turn>user\nQuestion: {ques}<end_of_turn>\n<start_of_turn>model\nAnswer:"
            else:
                ques = f"<start_of_turn>user\n{ques}<end_of_turn>\n<start_of_turn>model\n"
            
            if "model_output" in vector_dataset[i].keys():
                model_output = vector_dataset[i]["model_output"]
                ques += f" {model_output.strip()}"
            prompt_tokens_list.append(tokenizer.encode(ques, return_tensors="pt").to(device))
        else:
            raise NotImplementedError
    if args.trim == -1:
        if args.model_name == "gemma-2-9b-it":
            vector_root = os.path.join(args.data_path, args.data_name, "caa_vector_it", f"{args.model_name}_{args.mode}")
        elif args.model_name == "gemma-2-9b" or args.model_name=="llama-3.1":
            vector_root = os.path.join(args.data_path, args.data_name, "caa_vector_pt", f"{args.model_name}_{args.mode}")
        else:
            vector_root = os.path.join(args.data_path, args.data_name, "caa_vector", f"{args.model_name}_{args.mode}")
    else:
        if args.model_name == "gemma-2-9b-it":
            vector_root = os.path.join(args.data_path, args.data_name, "caa_vector_it", f"{args.model_name}_{args.mode}_trim")
        elif args.model_name == "gemma-2-9b" or args.model_name=="llama-3.1":
            vector_root = os.path.join(args.data_path, args.data_name, "caa_vector_pt", f"{args.model_name}_{args.mode}_trim")
        else:
            vector_root = os.path.join(args.data_path, args.data_name, "caa_vector", f"{args.model_name}_{args.mode}_trim")

    directory = os.path.dirname(args.output_file)  
    if not os.path.exists(directory):  
        os.makedirs(directory)  
    for layer in args.layers:
        print(f"Layer {layer}")
        if args.trim == -1:
            vector_path = os.path.join(
                vector_root, f"{layer}.pt"
            )
        else:
            vector_path = os.path.join(
                vector_root, f"{layer}_trim{args.trim}.pt"
            )
            print("vector_path: ", vector_path)

        steering_vector = torch.load(vector_path).to(device)
        print("Steering vector path: ",vector_path)
        print("Steering vector: ",steering_vector)
        print("Steering vector == 0: ", (steering_vector == 0).sum())
        print("Steering vector norm: ",steering_vector.norm())
        for multiplier in args.multipliers:
            
            preds = []
            preds_all = []
            model.reset_all()
            print(f"Multiplier {multiplier}")
            # print(f"##########steering_vector\n{steering_vector}##########")

            model.set_add_activations(
                layer, multiplier * steering_vector
            )
            max_new_tokens=args.max_new_tokens
            if "gsm" in args.eval_data_name:
                if args.model_name == "gemma-2-9b-it":
                    max_new_tokens=1024
                else:
                    max_new_tokens=512
            elif "exaggerated-safety" in args.eval_data_name:
                max_new_tokens=100
            elif "prefilling" in args.eval_data_name:
                max_new_tokens=100
            print("max_new_tokens: ", max_new_tokens)
            for prompt_tokens in tqdm(prompt_tokens_list, desc=f"Generating... layer-{layer} multiplied by {multiplier}"):
                prompt_tokens = prompt_tokens.to(device)  
                # output = model.model.generate(prompt_tokens, max_new_tokens=max_new_tokens)
                # print(f'##############output:\n{tokenizer.batch_decode(output)[0]}\n##############')
                # pdb.set_trace()
                if "gsm" in args.eval_data_name and (args.model_name=="gemma-2-9b" or args.model_name=="llama-3.1"):
                    stop_id_sequences=[tokenizer.encode("Question:", add_special_tokens=False)]
                    stopping_criteria=[KeyWordsCriteria(stop_id_sequences)]
                    output = model.model.generate(prompt_tokens, max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)
                else:
                    output = model.model.generate(prompt_tokens, max_new_tokens=max_new_tokens)
                preds_all.append(tokenizer.batch_decode(output)[0]) 
                output = output[:,prompt_tokens.shape[-1]:]
                output = tokenizer.batch_decode(output)
                preds.append(output[0])
            print("Without clean_preds!!!")
            # if "exaggerated-safety" not in args.eval_data_name:
            #     preds = clean_preds(preds)
            # preds = clean_preds(preds)
            if "toxigen" in args.eval_data_name:
                results = [
                    {"question": questions[idx], "pred": preds[idx], "label": label[idx]} for idx in range(len(preds))
                ]
            elif "gsm" in args.eval_data_name or "nqopen" in args.eval_data_name:
                results = [
                    {"question": questions[idx], "answer": answers[idx], "pred": preds[idx], "all": preds_all[idx]} for idx in range(len(preds))
                ]
            elif "exaggerated-safety" in args.eval_data_name:
                results = [
                    {"id":ids[idx], "type":tys[idx], "question": questions[idx], "pred": preds[idx]} for idx in range(len(preds))
                ]
            else:
                results = [
                    {"question": questions[idx], "pred": preds[idx], "all": preds_all[idx]} for idx in range(len(preds))
                ]
            output_file = args.output_file.replace(".result.json", f"_multiplier{multiplier}.result.json")

            json.dump(results, open(output_file, 'w'), indent=4, ensure_ascii=False)
            print(f"Output file: {output_file}")
                
