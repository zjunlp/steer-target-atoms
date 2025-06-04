import plotly.express as px
from typing import Any, Dict, Optional, Protocol, Tuple
import os, sys
import torch
from torch.utils.data import DataLoader
from sae_lens import SAE
from pathlib import Path
import numpy as np
from sae_lens.toolkit.pretrained_sae_loaders import (
    gemma_2_sae_loader,
    get_gemma_2_config,
)
from sae_lens import SAE, SAEConfig, LanguageModelSAERunnerConfig, SAETrainingRunner
import torch
import argparse
import pdb
import json
import tempfile
import shutil
import numpy as np
from safetensors import safe_open
from functools import partial
import math
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate
import random


def load_llama_sae(sae_dir: Path | str, device: str = "cpu") -> SAE:
    """
    Due to a bug (https://github.com/jbloomAus/SAELens/issues/168) in the SAE save implementation for SAE Lens we need to make
    a specialized workaround.

    WARNING this will be creating a directory where the files are LINKED with the exception of "cfg.json" which is copied. This is NOT efficient
    and you should not be calling it many times!

    This wraps: https://github.com/jbloomAus/SAELens/blob/main/sae_lens/sae.py#L284.

    SPECIFICALLY fix cfg.json.
    """
    sae_dir = Path(sae_dir)
    # print(f"Loading SAE from {sae_dir}")

    if not all([x.is_file() for x in sae_dir.iterdir()]):
        raise ValueError(
            "Not all files are present in the directory! Only files allowed for loading SAE Directory."
        )

    # https://github.com/jbloomAus/SAELens/blob/9dacd4a9672c138b7c900ddd9a28d1b3b3a0870c/sae_lens/config.py#L188
    # Load ourselves instead of from_json because there are some __dir__ elements that are not in the JSON
    # They should ALL be enumerated in `derivatives`
    ##### BEGIN #####
    cfg_f = sae_dir / "cfg.json"
    with open(cfg_f, "r") as f:
        cfg = json.load(f)
    derivatives = [
        "tokens_per_buffer",
    ]
    derivative_values = [cfg[x] for x in derivatives]
    for x in derivatives:
        del cfg[x]
    runner_config = LanguageModelSAERunnerConfig(**cfg)
    assert all(
        [
            d in runner_config.__dict__ and runner_config.__dict__[d] == dv
            for d, dv in zip(derivatives, derivative_values)
        ]
    )
    del derivative_values
    del derivatives
    ##### END #####

    # Load the SAE
    sae_config = runner_config.get_training_sae_cfg_dict()
    sae = None
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Copy in the CFG
        sae_config_f = temp_dir / "cfg.json"
        with open(sae_config_f, "w") as f:
            json.dump(sae_config, f)
        # Copy all the other files
        for name_f in sae_dir.iterdir():
            if name_f.name == "cfg.json":
                continue
            else:
                shutil.copy(name_f, temp_dir / name_f.name)
        # Load SAE
        sae = SAE.load_from_pretrained(temp_dir, device=device)
    assert sae is not None and isinstance(sae, SAE)

    with safe_open(os.path.join(sae_dir, "sparsity.safetensors"), framework="pt", device=device) as f:  # type: ignore
        log_sparsity = f.get_tensor("sparsity")

    return sae, log_sparsity


def load_gemma_sae(
    sae_path: str,
    device: str = "cpu",
    repo_id: str = "gemma-scope-9b-it-res",
    force_download: bool = False,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    d_sae_override: Optional[int] = None,
    layer_override: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Custom loader for Gemma 2 SAEs.
    """
    cfg_dict = get_gemma_2_config(repo_id, sae_path, d_sae_override, layer_override)
    cfg_dict["device"] = device

    # Apply overrides if provided
    if cfg_overrides is not None:
        cfg_dict.update(cfg_overrides)

    # Load and convert the weights
    state_dict = {}
    with np.load(os.path.join(sae_path, "params.npz")) as data:
        for key in data.keys():
            state_dict_key = "W_" + key[2:] if key.startswith("w_") else key
            state_dict[state_dict_key] = (
                torch.tensor(data[key]).to(dtype=torch.float32).to(device)
            )

    # Handle scaling factor
    if "scaling_factor" in state_dict:
        if torch.allclose(
            state_dict["scaling_factor"], torch.ones_like(state_dict["scaling_factor"])
        ):
            del state_dict["scaling_factor"]
            cfg_dict["finetuning_scaling_factor"] = False
        else:
            assert cfg_dict[
                "finetuning_scaling_factor"
            ], "Scaling factor is present but finetuning_scaling_factor is False."
            state_dict["finetuning_scaling_factor"] = state_dict.pop("scaling_factor")
    else:
        cfg_dict["finetuning_scaling_factor"] = False

    sae_cfg = SAEConfig.from_dict(cfg_dict)
    sae = SAE(sae_cfg)
    sae.load_state_dict(state_dict)

    # No sparsity tensor for Gemma 2 SAEs
    log_sparsity = None

    return sae, log_sparsity

def signed_min_max_normalize(tensor):
    abs_tensor = tensor.abs()
    min_val = abs_tensor.min()
    max_val = abs_tensor.max()
    normalized = (abs_tensor - min_val) / (max_val - min_val)
    return tensor.sign() * normalized  # 恢复正负符号

def act_and_fre(path_dir, 
                data_name,
                model_name,
                mode,
                select_type,
                layers,
                hook_module,
                trims,
                sae_path,
                re_error_way
                ):

    # path_dir=/disk3/wmr/ManipulateSAE/data/generation
    # if model_name == "llama-3.1":
    #     suffix = "ef16"
    # elif model_name == "gemma-2-9b":
    #     suffix = "16k"
    # else:
    #     raise ValueError("Precision not supported")
    if model_name == "gemma-2-9b-it":
        caa_vector_name = "caa_vector_it"
        sae_caa_vector_name = "sae_caa_vector_it"
    elif args.model_name == "gemma-2-9b":
        caa_vector_name = "caa_vector_pt"
        sae_caa_vector_name = "sae_caa_vector_pt"
    else:
        caa_vector_name = "caa_vector"
        sae_caa_vector_name = "sae_caa_vector"

    print(f're_error_way: {re_error_way}')
     
    for layer in layers:
        print(f'##########layer{layer}###########')
        print(f'##########layer{layer}###########')
        caa_vector = torch.load(f"{path_dir}/{data_name}/{caa_vector_name}/{model_name}_{mode}/{layer}.pt")

        print(f"no1-torch.norm(caa_vector): {torch.norm(caa_vector)}")
        for trim in trims:
            if args.model_name == "llama-3.1":
                suffix = "ef16"
                sae, _ = load_llama_sae(f"{sae_path}/layer_{layer}/{suffix}", device="cpu")
            elif args.model_name == "gemma-2-9b" or args.model_name == "gemma-2-9b-it":
                print(f'type(sae_path)\n{type(sae_path)}\n\n{sae_path}')
                suffix = "16k"
                sae, _ = load_gemma_sae(sae_path, device="cpu")
                # sae, _ = load_gemma_sae(args.sae_path, device="cpu")
        # 激活和频率取前pec的交集
            neg_attr_name=f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/sae_vector/feature_attr/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_neg_feature_freq.pt"
            pos_attr_name=f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/sae_vector/feature_attr/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_pos_feature_freq.pt"
            act_attr_name=f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/sae_vector/feature_attr/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_feature_score.pt"
            neg_data = torch.load(neg_attr_name)
            pos_data = torch.load(pos_attr_name)
            act_data = torch.load(act_attr_name)

            act_data_init = act_data.to(sae.W_dec.device)
            re_error = act_data_init @ sae.W_dec

            print(f"act_data.norm：{torch.norm(act_data)}")
            print(f"re_error: {torch.norm(re_error)}\n\ntop 10:{torch.sort(re_error, descending=True)[0][:10]}")
            print(f"Negative data:{torch.norm(neg_data)} \n\ntop 10:{torch.sort(neg_data, descending=True)[0][:10]}")
            print(f"Positive data:{torch.norm(pos_data)} \n\ntop 10:{torch.sort(pos_data, descending=True)[0][:10]}")
            print(f"Activation data:{torch.norm(act_data)} \n\ntop 10:{torch.sort(act_data, descending=True)[0][:10]}" )


            #设置剪枝比例
            pec = trim

            diff_data = pos_data - neg_data

            # 1. Min-Max归一化，保留正负符号

            norm_act = signed_min_max_normalize(act_data)  # 激活值差值归一化
            norm_diff = signed_min_max_normalize(diff_data)   # 激活频率差值归一化

            # 2. 符号一致性筛选
            mask = ((norm_act > 0) & (norm_diff > 0)) | ((norm_act < 0) & (norm_diff < 0))
            print("mask:",mask.sum())

            # 3. 综合得分计算（乘积方法）
            scores = torch.zeros_like(norm_diff)  # 初始化综合得分
            scores[mask] = (norm_diff[mask])  # 仅计算符号一致的维度得分

            print("act_data != 0: ", (act_data != 0).sum())
            print("freq_scores != 0: ", (scores != 0).sum())
            # # 4. 筛选前pec激活位置
            # topk_percent_indices = torch.argsort(torch.abs(scores), descending=True, stable=True)[:int(pec * len(scores))+1]  # 按得分降序取前pec

            # # 5. 创建掩码（基于 top_5_percent_indices）
            # prune_mask = torch.zeros_like(act_data, dtype=torch.bool)  # 初始化掩码
            # prune_mask[topk_percent_indices] = True  # 前5%的位置设为True

            threshold_fre = torch.sort(torch.abs(scores), descending=True, stable=True).values[int(pec * len(scores))]
            print(f'频率阈值: {threshold_fre}')
            prune_mask = torch.abs(scores) >= threshold_fre
            print("prune_mask:",prune_mask.sum())


            act_data_combined = act_data.clone()
            ######### act and fre ########
            print(f"######### act and fre ########")
            threshold = torch.sort(torch.abs(act_data_combined), descending=True, stable=True).values[int(pec * len(act_data_combined))]
            print(f'阈值: {threshold}')
            act_top_mask = torch.abs(act_data_combined) >= threshold
            print("act_top_mask:",act_top_mask.sum())

            # combined_mask = prune_mask | act_top_mask
            combined_mask = prune_mask & act_top_mask
            print("combined_mask:",combined_mask.sum())
            act_data_combined[~combined_mask] = 0
            print(torch.abs(act_data_combined).sum())

            act_data_combined = act_data_combined.to(sae.W_dec.device)
            result_combined = act_data_combined @ sae.W_dec
            print("result_combined.shape",result_combined.shape)
            print("result_combined:",result_combined)
            print("torch.norm(result_combined)", torch.norm(result_combined))

            print(f"########### only act ########")
            ########### only act ########
            act_data_act = act_data.clone()
            act_threshold = torch.sort(torch.abs(act_data_act), descending=True, stable=True).values[int(pec * len(act_data_act))]
            print(f'阈值: {act_threshold}')
            act_mask = torch.abs(act_data_act) >= act_threshold
            print("act_top_mask:",act_top_mask.sum())
            act_data_act[~act_mask] = 0
            act_data_act = act_data_act.to(sae.W_dec.device)
            result_act = act_data_act @ sae.W_dec
            print("result_act.shape:",result_act.shape)
            print("result_act:",result_act)


            ########### only fre ########
            print(f"########### only fre ########")
            act_data_fre = act_data.clone()
            act_data_fre[~prune_mask] = 0
            act_data_fre = act_data_fre.to(sae.W_dec.device)
            result_fre = act_data_fre @ sae.W_dec
            print("result_fre.shape:",result_fre.shape)
            print("result_fre:",result_fre)
            
            if re_error_way:
                result_combined += re_error
                result_act += re_error
                result_fre += re_error
                steering_vector_act_and_fre = f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim_re_error/steering_vector/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_act_and_fre_trim{pec}.pt"
                steering_vector_act = f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim_re_error/steering_vector/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_act_trim{pec}.pt"
                steering_vector_fre = f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim_re_error/steering_vector/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_fre_trim{pec}.pt"
            else:
                steering_vector_act_and_fre = f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim/steering_vector/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_act_and_fre_trim{pec}.pt"
                steering_vector_act = f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim/steering_vector/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_act_trim{pec}.pt"
                steering_vector_fre = f"{path_dir}/{data_name}/{sae_caa_vector_name}/{model_name}_{mode}/act_and_fre_trim/steering_vector/{model_name}_sae_layer{layer}_{hook_module}_{suffix}_fre_trim{pec}.pt"

        

            
            parent_dir = os.path.dirname(steering_vector_act_and_fre)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            torch.save(result_combined, steering_vector_act_and_fre)
            torch.save(result_act, steering_vector_act)
            torch.save(result_fre, steering_vector_fre)

# 频率top

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dir", type=str, default="dir")
    parser.add_argument("--sae_path", type=str, default="dir")
    parser.add_argument("--data_name", type=str, default="power-seeking")
    parser.add_argument("--model_name", type=str, default="llama-3.1")
    parser.add_argument("--mode", type=str, default="toxic")
    parser.add_argument("--select_type", type=str, default="sae_vector")
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--hook_module", default="resid_post")
    parser.add_argument("--trim", nargs="+", type=float)
    parser.add_argument("--re_error_way", action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    # print(args.layers)
    # pdb.set_trace()

    # sae_path = "/mnt/20t/msy/models/gemma-scope-9b-it-res/layer_20/width_131k/average_l0_24"
    # sae, _ = load_gemma_2_sae(sae_path=sae_path, device="cuda:5")
    # llama_sae, _ = load_sae_from_dir("/mnt/20t/msy/shae/exp/llama-3.1-jumprelu-resid_post/layer_20/ef16")

    

    act_and_fre(args.path_dir, 
                args.data_name,
                args.model_name,
                args.mode,
                args.select_type,
                args.layers,
                args.hook_module,
                args.trim,
                args.sae_path,
                args.re_error_way
                )
