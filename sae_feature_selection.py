import os, sys
import pdb
import torch
from torch.utils.data import DataLoader
from sae_lens import SAE
from pathlib import Path
import json
import math
from tqdm import tqdm
from datasets import load_dataset
import random
import argparse
from sae_utils import (
    attribution_patching,
    activation_selection,
    activation_selection_contrastive_for_only_sys,
    activation_selection_contrastive_for_toxic_freq,
    activation_selection_contrastive_for_toxic_freq_AB,
    clear_gpu_cache,
    load_sae_from_dir,
    load_gemma_2_sae,
    generate_sae_steering_vector
)
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
)
from dataloader import (
    GenerationDataset, 
    YNDataset, 
    PersonalityEditDataset,
    SafetyDataset
)
from transformers import AutoTokenizer
from baseline.caa.utils.input_format import llama3_chat_input_format

import pdb

import random
random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--select_type",
        default="sae_vector",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        default="./test.json",
        type=str,
    )
    parser.add_argument(
        "--data_file",
        default="./test.json",
        type=str,
    )
    parser.add_argument(
        "--data_name",
        default="power-seeking",
        type=str,
    )
    parser.add_argument(
        "--model_name_or_path",
        default="./",
        type=str,
    )
    parser.add_argument(
        "--sae_path",
        default="./",
        type=str,
    )
    parser.add_argument(
        "--mode",
        default="personality",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        default="llama-3.1-8b-insturct",
        type=str,
    )
    parser.add_argument(
        "--system_prompt",
        default="",
        type=str,
    )
    parser.add_argument(
        "--steering_vector_name",
        default="steering_vector.pt",
        type=str,
    )
    parser.add_argument("--AB", action="store_true", default=False)


    args = parser.parse_args()
    print(args)

    device = "cuda:0"
    if "gemma" in args.model_name_or_path.lower():
        sae, sparsity = load_gemma_2_sae(args.sae_path, device=device)
    else:
        sae, sparsity = load_sae_from_dir(args.sae_path, device=device)

    model = HookedTransformer.from_pretrained(
        args.model_name_or_path,
        device=device,
        # torch_dtype=torch.float32,
        # torch_dtype=torch.bfloat16,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # set select fucntion
    if args.select_type == "atp":
        selection_func = attribution_patching
    elif args.select_type == "act":
        selection_func = activation_selection
    elif args.select_type == "only_sys":
        selection_func = activation_selection_contrastive_for_only_sys
    elif args.select_type == "sae_vector":
        selection_func = activation_selection_contrastive_for_toxic_freq
    elif args.select_type == "sae_vector_AB":
        selection_func = activation_selection_contrastive_for_toxic_freq_AB
    else:
        raise NotImplementedError
    selection_record = {}

    prompts, neg_prompts, prefix_prompts = [], [], []

    

    if args.mode == "personality" or args.mode == "safety" or args.mode=="PersonaMyopic" or args.mode=="PersonaSurvival" or args.mode=="PersonaSycophancy" or args.mode=="PersonaHallucination":
        print(f'#########mode:{args.mode}#############')

        dataset = GenerationDataset()
        selection_data = dataset.get_data_for_selection(
            data_file=args.data_file,
        )

        if args.select_type == "sae_vector":

            for i in range(len(selection_data)):
            # ques = "<start_of_turn>user\nQurstion: " + vector_dataset[i]["question"] + "\n<start_of_turn>model"
                if args.AB == True and args.model_name=="gemma-2-9b-it":
                    ques = "<start_of_turn>user\nQuestion: "+ selection_data[i]["question"] + "<end_of_turn>\n<start_of_turn>model"
                    chosen = "\nAnswer:" + selection_data[i]["chosen"]
                    rejected = "\nAnswer:" + selection_data[i]["rejected"]
                elif args.AB == True and args.model_name=="gemma-2-9b":
                    ques = "Question: "+ selection_data[i]["question"]
                    chosen = "\nAnswer:" + selection_data[i]["chosen"]
                    rejected = "\nAnswer:" + selection_data[i]["rejected"]
                else:
                    ques = selection_data[i]["question"]
                    chosen = selection_data[i]["chosen"]
                    rejected = selection_data[i]["rejected"]

                if args.model_name=="gemma-2-9b":
                    if args.system_prompt != "":
                        if ques is not None:
                            ques = args.system_prompt + " " + ques
                        else:
                            ques = args.system_prompt
                # # 生成vector的input_forma体现在--args.AB
                elif args.model_name=="gemma-2-9b-it" and args.AB == False:
                    if args.system_prompt != "":
                        print('system_prompt for gemma-2-9b-it is not done')
                        raise NotImplementedError
                    else:
                        ques = f"<start_of_turn>user\n{ques}<end_of_turn>\n<start_of_turn>model\n"
                        print("ques:", ques)
                else:
                    raise NotImplementedError
                
                
                # for constrastive selection within act
                if ques and chosen and rejected:
                    prompts.append({
                        "ques": ques,
                        "pos": ques + chosen,
                        "neg": ques + rejected,
                    })

        elif args.select_type == "only_sys":
            for i in range(len(selection_data)):
                ques = selection_data[i]["question"] # string
                chosen = selection_data[i]["chosen"]
                rej = selection_data[i]["rejected"]
                assert chosen == rej , "Error Chosen and Rej!!!"
                print("chose: ", chosen)

                if args.model_name=="gemma-2-9b-it":
                    if args.system_prompt != "":
                        print('system_prompt for gemma-2-9b-it is not done')
                        raise NotImplementedError
                    else:
                        ques_chosen = f"<start_of_turn>user\n{ques}<end_of_turn>\n<start_of_turn>model\n"
                        ques_rej = f"<start_of_turn>user\n<end_of_turn>\n<start_of_turn>model\n"
                        # print("ques:", ques)
                # for constrastive selection within act
                prompts.append({
                    "ques_chosen": ques_chosen,
                    "ques_rej": ques_rej,
                    "pos": ques_chosen + chosen,
                    "neg": ques_rej + rej,
                })
        else:
            raise NotImplementedError 

    else:
        raise NotImplementedError
    
    if args.select_type == "only_sys":
        feature_score, pos_feature_freq, neg_feature_freq, pos_act_mean, neg_act_mean = selection_func(
            model=model,
            sae=sae,
            prompts=prompts,
            neg_prompts=neg_prompts if len(neg_prompts) > 0 else None,
            prefix_prompts=prefix_prompts if len(prefix_prompts) > 0 else None,
            batch_size=args.batch_size,
            model_name=args.model_name,
            desc=args.mode,
            model_name_or_path=args.model_name_or_path,
        )

        output_dir = os.path.dirname(args.output_file)
        steering_vector_dir = os.path.join(output_dir, "steering_vector")
        feature_attr_dir = os.path.join(output_dir, "feature_attr")
        act_mean_dir = os.path.join(output_dir, "act_mean")
        if not os.path.exists(steering_vector_dir):
            os.makedirs(steering_vector_dir)
        if not os.path.exists(feature_attr_dir):
            os.makedirs(feature_attr_dir)
        if not os.path.exists(act_mean_dir):
            os.makedirs(act_mean_dir)
        
        steering_vector_path = os.path.join(steering_vector_dir, args.steering_vector_name)
        steering_vector = feature_score @ sae.W_dec
        # print(f'feature_score:{feature_score}\n\nsae.W_dec{sae.W_dec}\n\nsteering_vector{steering_vector}')
        # pdb.set_trace()

        print("steering_vector.shape:", steering_vector.shape)
        print("steering_vector:", steering_vector)
        print("steering_vector.norm:", steering_vector.norm())
        torch.save(steering_vector, steering_vector_path)

        feature_attr_name = args.steering_vector_name
        if not feature_attr_name.endswith('steering_vector.pt'):
            feature_attr_name += '_steering_vector.pt'
        feature_attr_name = feature_attr_name.replace('steering_vector.pt', 'feature_score.pt')
        feature_attr_path = os.path.join(feature_attr_dir, feature_attr_name)
        print("feature_attr.shape:", feature_score.shape)
        torch.save(feature_score, feature_attr_path)

        pos_feature_freq_path = feature_attr_name.replace('feature_score.pt', 'pos_feature_freq.pt')
        pos_feature_freq_path = os.path.join(feature_attr_dir, pos_feature_freq_path)
        print("pos_feature_freq.shape:", pos_feature_freq.shape)
        torch.save(pos_feature_freq, pos_feature_freq_path)

        neg_feature_freq_path = feature_attr_name.replace('feature_score.pt', 'neg_feature_freq.pt')
        neg_feature_freq_path = os.path.join(feature_attr_dir, neg_feature_freq_path)
        print("neg_feature_freq.shape:", neg_feature_freq.shape)
        torch.save(neg_feature_freq, neg_feature_freq_path)
        
        pos_act_mean_path = feature_attr_name.replace('feature_score.pt', 'pos_act_mean.pt')
        pos_act_mean_path = os.path.join(act_mean_dir, pos_act_mean_path)
        print("pos_act_mean.shape:", pos_act_mean.shape)
        torch.save(pos_act_mean, pos_act_mean_path)

        neg_act_mean_path = feature_attr_name.replace('feature_score.pt', 'neg_act_mean.pt')
        neg_act_mean_path = os.path.join(act_mean_dir, neg_act_mean_path)
        print("neg_act_mean.shape:", neg_act_mean.shape)
        torch.save(neg_act_mean, neg_act_mean_path)
    elif args.select_type == "sae_vector" or args.select_type == "sae_vector_AB":
        feature_score, pos_feature_freq, neg_feature_freq, pos_act_mean, neg_act_mean = selection_func(
            model=model,
            sae=sae,
            prompts=prompts,
            neg_prompts=neg_prompts if len(neg_prompts) > 0 else None,
            prefix_prompts=prefix_prompts if len(prefix_prompts) > 0 else None,
            batch_size=args.batch_size,
            model_name=args.model_name,
            desc=args.mode,
            model_name_or_path=args.model_name_or_path,
        )

        output_dir = os.path.dirname(args.output_file)
        steering_vector_dir = os.path.join(output_dir, "steering_vector")
        feature_attr_dir = os.path.join(output_dir, "feature_attr")
        act_mean_dir = os.path.join(output_dir, "act_mean")
        if not os.path.exists(steering_vector_dir):
            os.makedirs(steering_vector_dir)
        if not os.path.exists(feature_attr_dir):
            os.makedirs(feature_attr_dir)
        if not os.path.exists(act_mean_dir):
            os.makedirs(act_mean_dir)
        
        steering_vector_path = os.path.join(steering_vector_dir, args.steering_vector_name)
        steering_vector = feature_score @ sae.W_dec
        # print(f'feature_score:{feature_score}\n\nsae.W_dec{sae.W_dec}\n\nsteering_vector{steering_vector}')
        # pdb.set_trace()

        print("steering_vector.shape:", steering_vector.shape)
        print("steering_vector:", steering_vector)
        print("steering_vector.norm:", steering_vector.norm())
        torch.save(steering_vector, steering_vector_path)

        feature_attr_name = args.steering_vector_name
        if not feature_attr_name.endswith('steering_vector.pt'):
            feature_attr_name += '_steering_vector.pt'
        feature_attr_name = feature_attr_name.replace('steering_vector.pt', 'feature_score.pt')
        feature_attr_path = os.path.join(feature_attr_dir, feature_attr_name)
        print("feature_attr.shape:", feature_score.shape)
        torch.save(feature_score, feature_attr_path)

        pos_feature_freq_path = feature_attr_name.replace('feature_score.pt', 'pos_feature_freq.pt')
        pos_feature_freq_path = os.path.join(feature_attr_dir, pos_feature_freq_path)
        print("pos_feature_freq.shape:", pos_feature_freq.shape)
        torch.save(pos_feature_freq, pos_feature_freq_path)

        neg_feature_freq_path = feature_attr_name.replace('feature_score.pt', 'neg_feature_freq.pt')
        neg_feature_freq_path = os.path.join(feature_attr_dir, neg_feature_freq_path)
        print("neg_feature_freq.shape:", neg_feature_freq.shape)
        torch.save(neg_feature_freq, neg_feature_freq_path)
        
        pos_act_mean_path = feature_attr_name.replace('feature_score.pt', 'pos_act_mean.pt')
        pos_act_mean_path = os.path.join(act_mean_dir, pos_act_mean_path)
        print("pos_act_mean.shape:", pos_act_mean.shape)
        torch.save(pos_act_mean, pos_act_mean_path)

        neg_act_mean_path = feature_attr_name.replace('feature_score.pt', 'neg_act_mean.pt')
        neg_act_mean_path = os.path.join(act_mean_dir, neg_act_mean_path)
        print("neg_act_mean.shape:", neg_act_mean.shape)
        torch.save(neg_act_mean, neg_act_mean_path)

    else:
        values, _values = selection_func(
            model=model,
            sae=sae,
            prompts=prompts,
            neg_prompts=neg_prompts if len(neg_prompts) > 0 else None,
            prefix_prompts=prefix_prompts if len(prefix_prompts) > 0 else None,
            batch_size=args.batch_size,
            model_name=args.model_name_or_path,
            desc=args.mode,
        )

        _, indices = torch.topk(_values, k=1000)

        values_recons = torch.zeros_like(values)
        values_recons[indices] = values[indices]
        steering_vector_recons = sae.decode(values_recons)
        
        values = values[indices]
        steering_vector_direct = (sae.W_dec[indices] * values.unsqueeze(-1)).sum(dim=0)

        selection_record[args.data_name] = {
            "values": values.cpu().numpy().tolist(),
            "indices": indices.cpu().numpy().tolist(),
        }

        output_dir = os.path.dirname(args.output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(
            steering_vector_direct,
            args.output_file+".steering_vector_direct.pt",
        )
        
        torch.save(
            steering_vector_recons,
            args.output_file+".steering_vector_recons.pt",
        )

        json.dump(
            selection_record,
            open(args.output_file, "w"),
            indent=4,
            ensure_ascii=False,
        )
