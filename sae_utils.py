import plotly.express as px
from typing import Any, Dict, Optional, Protocol, Tuple
import tempfile
import os, sys
import torch
from torch.utils.data import DataLoader
from sae_lens import SAE
from pathlib import Path
import json
import shutil
import numpy as np
from sae_lens.toolkit.pretrained_sae_loaders import (
    gemma_2_sae_loader,
    get_gemma_2_config,
)
from sae_lens import SAE, SAEConfig, LanguageModelSAERunnerConfig, SAETrainingRunner
from safetensors import safe_open
from functools import partial
import math
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate
import random
# from baseline.caa.utils.input_format import llama3_chat_input_format

from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

sys.path.append("../")
import einops
import pdb

# fix the seed for reproducing atp selection
random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

def extend_attention_mask_by_fixed_len(attention_mask, x):
    '''
    Args:
        attention_mask: torch.Tensor, (bsz, seq_len)
        x: int
    
    Returns:
        torch.Tensor
    '''
    bsz, seq_len = attention_mask.shape
    for i in range(bsz):
        last_zero_index = (attention_mask[i] == 0).nonzero(as_tuple=True)[0]

        if last_zero_index.numel() > 0:
            last_zero_index = last_zero_index[-1].item()
        else:
            last_zero_index = -1

        end_index = min(last_zero_index + x + 1, attention_mask.shape[1])
        attention_mask[i, last_zero_index + 1 : end_index] = 0

    return attention_mask


def extend_attention_mask_by_lens(attention_mask, x):
    '''
    Args:
        attention_mask: torch.Tensor, (bsz, seq_len)
        x: List[int]
    
    Returns:
        torch.Tensor
    '''
    bsz, seq_len = attention_mask.shape
    for i in range(bsz):
        last_zero_index = (attention_mask[i] == 0).nonzero(as_tuple=True)[0]

        if last_zero_index.numel() > 0:
            last_zero_index = last_zero_index[-1].item()
        else:
            last_zero_index = -1

        end_index = min(last_zero_index + x[i] + 1, attention_mask.shape[1])
        attention_mask[i, last_zero_index + 1 : end_index] = 0

    return attention_mask

def load_sae_from_dir(sae_dir: Path | str, device: str = "cpu") -> SAE:
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


def clear_gpu_cache(cache):
    del cache
    torch.cuda.empty_cache()


def get_cache_and_logits(model, tokens, attention_mask, sae, add_noise=False):

    # add hook for sae, to get the gradient of sae modules
    filter_not_input = lambda name: "_input" not in name

    grad_cache = {}
    cache = {}

    def sae_bwd_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    def sae_fwd_hook(act, hook):
        cache[hook.name] = act.detach()

    sae.reset_hooks()
    sae.add_hook(filter_not_input, sae_fwd_hook, "fwd")
    sae.add_hook(filter_not_input, sae_bwd_hook, "bwd")

    # add hook for model, to replace the original mlp output with the sae reconstruction.
    def reconstr_direct(hidden_states, hook, attention_mask, add_noise):

        if add_noise:

            # Step 1: Compute the mean and variance of the hidden states
            expanded_attention_mask = attention_mask.unsqueeze(-1)  # (bsz, seq, -1)
            valid_hidden_states = (
                hidden_states * expanded_attention_mask
            )  # (bsz, seq, hidden_size)
            valid_token_count = attention_mask.sum(dim=1, keepdim=True)  # (bsz, 1)

            hidden_mean = valid_hidden_states.sum(dim=1) / valid_token_count
            hidden_var = (
                (valid_hidden_states - hidden_mean.unsqueeze(1)) ** 2
                * expanded_attention_mask
            ).sum(dim=1) / valid_token_count

            hidden_std = torch.sqrt(hidden_var)

            # Step 2: Sampling Noise, Sampleing From ~ N(mean, std*3)
            noise = torch.randn_like(hidden_states) * (
                3 * hidden_std.unsqueeze(1)
            ) + hidden_mean.unsqueeze(1)

            # Step 3: Add Noise to the hidden states
            hidden_states += noise * expanded_attention_mask

        output = sae(hidden_states)
        return output

    model.reset_hooks()

    direct_output = model.run_with_hooks(
        tokens,
        attention_mask=attention_mask,
        fwd_hooks=[
            (
                sae.cfg.hook_name,
                partial(
                    reconstr_direct, attention_mask=attention_mask, add_noise=add_noise
                ),
            ),
        ],
        return_type="loss",
    )

    direct_output.backward()

    sae.reset_hooks()
    model.reset_hooks()

    return (direct_output.item(), cache, grad_cache)


def get_cache_and_logits_act(model, tokens, attention_mask, sae):
    '''
    get the activation cache for all sae modules
    dict_keys(['hook_sae_acts_pre', 'hook_sae_acts_post', 'hook_sae_recons', 'hook_sae_output'])
    Return: cache: Dict[str, torch.Tensor]
    '''
    filter_not_input = lambda name: "_input" not in name
    cache = {}

    def sae_fwd_hook(act, hook):
        cache[hook.name] = act.detach()

    sae.reset_hooks()
    sae.add_hook(filter_not_input, sae_fwd_hook, "fwd")

    # add hook for model, to replace the original mlp output with the sae reconstruction.
    def reconstr_direct(activations, hook):
        cache["activations"] = activations.detach()
        output = sae(activations)
        return output

    model.reset_hooks()

    direct_output = model.run_with_hooks(
        tokens,
        attention_mask=attention_mask,
        fwd_hooks=[
            (
                sae.cfg.hook_name,
                partial(reconstr_direct),
            ),
        ],
        return_type="loss",
    )

    sae.reset_hooks()
    model.reset_hooks()
    return cache


def attr_patch_sae(
    clean_cache, corrupted_cache, corrputed_grad_cache, hook_name, attention_mask
):
    pos_logit = clean_cache[hook_name]
    neg_logit = corrupted_cache[hook_name]
    neg_grad = corrputed_grad_cache[hook_name]

    feature_attr = neg_grad * (pos_logit - neg_logit)
    feature_attr = feature_attr * attention_mask.unsqueeze(-1)
    feature_attr = einops.reduce(
        feature_attr,
        "batch pos feature_nums -> batch feature_nums",
        "sum",
    )
    feature_attr = einops.reduce(
        feature_attr,
        "batch feature_nums -> feature_nums",
        "mean",
    )

    return feature_attr


def attribution_patching(
    model, 
    sae, 
    prompts,
    neg_prompts=None,
    prefix_texts=None, 
    batch_size=4, 
    model_name=None, 
    desc="None",
    mean=True
):
    '''
    Return:
        total_feature_attr: tensor of shape (feature_nums, weights)
    '''
    indice = len(prompts)

    shuffled_indices = list(range(indice))

    random.shuffle(shuffled_indices)

    total_feature_attr = []

    for i in tqdm(range(0, indice, batch_size), desc=f"atp for {desc}"):
        batch_indices = shuffled_indices[i : i + batch_size]
        batch_prompt = [prompts[i] for i in batch_indices]

        tokens = model.to_tokens(
            batch_prompt,
            padding_side="right" if "gemma" in model_name else "left",
        )
        attention_mask = torch.ones_like(tokens)
        attention_mask[tokens == model.tokenizer.pad_token_id] = 0

        clean_value, clean_cache, clean_grad_cache = get_cache_and_logits(
            model, tokens, attention_mask, sae, add_noise=False
        )

        clear_gpu_cache(clean_grad_cache)

        corrupted_value, corrupted_cache, corrputed_grad_cache = get_cache_and_logits(
            model, tokens, attention_mask, sae, add_noise=True
        )

        feature_attr = attr_patch_sae(
            clean_cache=clean_cache,
            corrupted_cache=corrupted_cache,
            corrputed_grad_cache=corrputed_grad_cache,
            hook_name="hook_sae_acts_post",
            attention_mask=attention_mask,
        )

        total_feature_attr.append(feature_attr)

        clear_gpu_cache(clean_value)
        clear_gpu_cache(corrupted_value)
        clear_gpu_cache(clean_cache)
        clear_gpu_cache(corrupted_cache)
        clear_gpu_cache(corrputed_grad_cache)

    total_feature_attr = torch.stack(total_feature_attr).mean(0)

    return total_feature_attr, total_feature_attr


def generate_sae_steering_vector(
    model,
    sae,
    feature_idxs,
):
    '''
    Args:
        model: huggingface model
        sae: SAE model
        feature_idxs: List[int], the selected feature index for the steering vector
    '''
    import pdb;pdb.set_trace()

    return sae.steering_vector


def activation_selection(
    model, 
    sae, 
    prompts,
    neg_prompts=None,
    prefix_prompts=None, 
    batch_size=4, 
    model_name=None, 
    desc="None",
    mean=True
):
    '''
    Args:
        model: huggingface model
        sae: SAE model
        prompts: List[str], positive text
        neg_prompts: List[str], negative text, default is None, if not None, computing the contrastive activation
        prefix_prompts: List[str], prefix text, for filtering out the token ids before computing the contrastive activation,
                        defualt is None, assume that the prefix_text for pos and neg are the same.
        batch_size: batch size
        model_name: model name
        desc: description
    '''

    indice = len(prompts)

    shuffled_indices = list(range(indice))
    random.shuffle(shuffled_indices)

    prefix_lens = [0 for _ in range(len(prompts))]
    if prefix_prompts is not None:
        for i in range(len(prefix_prompts)):
            prefix_tokens = model.to_tokens(
                prefix_prompts[i],
            )
            prefix_lens[i] = prefix_tokens.shape[-1]

    def get_feature_acts(prompt):
        tokens = model.to_tokens(
            prompt,
            padding_side="right" if "gemma" in model_name else "left",
        )
        attention_mask = torch.ones_like(tokens)
        attention_mask[tokens == model.tokenizer.pad_token_id] = 0

        cache = get_cache_and_logits_act(model, tokens, attention_mask, sae)

        feature_acts = cache["hook_sae_acts_post"]
        # extend the attention mask to include the prefix tokens
        attention_mask = extend_attention_mask_by_lens(attention_mask, prefix_lens)
        feature_acts = feature_acts * attention_mask.unsqueeze(-1)

        feature_acts = einops.reduce(
            feature_acts,
            "batch pos feature_nums -> batch feature_nums",
            "mean",
        )

        return feature_acts

    feature_acts_list, neg_feature_acts_list = [], []

    for i in tqdm(range(0, indice, batch_size), desc=f"act for {desc}", disable=not mean):
        batch_indices = shuffled_indices[i : i + batch_size]
 
        prompt = [prompts[i] for i in batch_indices]
        feature_acts = get_feature_acts(prompt)
        feature_acts_list.append(feature_acts)

        if neg_prompts is not None:
            neg_prompt = [neg_prompts[i] for i in batch_indices]
            neg_feature_acts = get_feature_acts(neg_prompt)
            neg_feature_acts_list.append(neg_feature_acts)
    
    if neg_prompts is not None:
        pos_feature_acts = torch.cat(feature_acts_list, dim=0)
        neg_feature_acts = torch.cat(neg_feature_acts_list, dim=0)
        feature_score = (
            pos_feature_acts.mean(0) - neg_feature_acts.mean(0)
        )
        return feature_score, torch.abs(feature_score)
    
    feature_acts = torch.cat(feature_acts_list, dim=0)

    if not mean:
        # for selecting acitvation feature for each prompt
        return feature_acts

    return feature_score, torch.abs(feature_score)

def activation_selection_contrastive_for_toxic_freq(
    model, 
    sae, 
    prompts,
    neg_prompts=None,
    prefix_prompts=None, 
    batch_size=4, 
    model_name=None, 
    desc="None",
    mean=True,
    model_name_or_path="None",
):
    print(f"Computing the activation selection for activation_selection_contrastive_for_toxic_freq")
    indice = len(prompts)

    shuffled_indices = list(range(indice))
    random.shuffle(shuffled_indices)

    ques_feature_acts_list = []
    pos_feature_acts_list = []
    neg_feature_acts_list = []

    for i in tqdm(shuffled_indices, desc=f"act for {desc}"):
        # try:
            ques = prompts[i]["ques"]
            # if "llama-3.1-8b-instruct" in model_name_or_path.lower():
            #     ques = llama3_chat_input_format(ques, model_output=None, system_prompt=None)
            pos_prompt = prompts[i]["pos"]
            neg_prompt = prompts[i]["neg"]

            ques_tokens = model.to_tokens(
                ques,
                padding_side="right" if "gemma" in model_name else "left",
            )

            ques_tokens_len = ques_tokens.shape[-1]

            pos_tokens = model.to_tokens(
                pos_prompt,
                padding_side="right" if "gemma" in model_name else "left",
            )
            pos_attention_mask = torch.ones_like(pos_tokens)
            pos_attention_mask[pos_tokens == model.tokenizer.pad_token_id] = 0

            pos_cache = get_cache_and_logits_act(model, pos_tokens, pos_attention_mask, sae)

            pos_feature_acts = pos_cache["hook_sae_acts_post"]
            pos_feature_acts = pos_feature_acts * pos_attention_mask.unsqueeze(-1)

            ques_feature_acts = pos_feature_acts[:, :ques_tokens_len, :]
            pos_feature_acts = pos_feature_acts[:, ques_tokens_len:, :]

            ques_feature_acts = einops.reduce(
                ques_feature_acts,
                "batch ques feature_nums -> batch feature_nums",
                "mean",
            )

            pos_feature_acts = einops.reduce(
                pos_feature_acts,
                "batch pos feature_nums -> batch feature_nums",
                "mean",
            )

            neg_tokens = model.to_tokens(
                neg_prompt,
                padding_side="right" if "gemma" in model_name else "left",
            )
            neg_attention_mask = torch.ones_like(neg_tokens)
            neg_attention_mask[neg_tokens == model.tokenizer.pad_token_id] = 0

            neg_cache = get_cache_and_logits_act(model, neg_tokens, neg_attention_mask, sae)

            neg_feature_acts = neg_cache["hook_sae_acts_post"]
            neg_feature_acts = neg_feature_acts * neg_attention_mask.unsqueeze(-1)
            neg_feature_acts = neg_feature_acts[:, ques_tokens_len:, :]

            neg_feature_acts = einops.reduce(
                neg_feature_acts,
                "batch neg feature_nums -> batch feature_nums",
                "mean",
            )

            ques_feature_acts_list.append(ques_feature_acts)
            pos_feature_acts_list.append(pos_feature_acts)
            neg_feature_acts_list.append(neg_feature_acts)
        # except RuntimeError as e:
        #     if "out of memory" in str(e):
        #         print(f"Skipping due to OOM at index {i}")
        #         torch.cuda.empty_cache()
        #     else:
        #         raise e
    ques_feature_acts = torch.cat(ques_feature_acts_list, dim=0)
    pos_feature_acts = torch.cat(pos_feature_acts_list, dim=0)
    neg_feature_acts = torch.cat(neg_feature_acts_list, dim=0)
    # pdb.set_trace()
    feature_score = (
        pos_feature_acts.mean(0) - neg_feature_acts.mean(0)
    )

    pos_feature_freq = (pos_feature_acts > 0).float().sum(0)
    print(pos_feature_freq.shape)
    print(pos_feature_freq)
    neg_feature_freq = (neg_feature_acts > 0).float().sum(0)
    print(neg_feature_freq.shape)
    print(neg_feature_freq)

    pos_act_mean = pos_feature_acts.mean(0)
    print(pos_act_mean.shape)
    print(pos_act_mean)
    neg_act_mean = neg_feature_acts.mean(0)
    print(neg_act_mean.shape)
    print(neg_act_mean)

    return feature_score, pos_feature_freq, neg_feature_freq, pos_act_mean, neg_act_mean

def activation_selection_contrastive_for_toxic_freq_AB(
    model, 
    sae, 
    prompts,
    neg_prompts=None,
    prefix_prompts=None, 
    batch_size=4, 
    model_name=None, 
    desc="None",
    mean=True
):
    print(f"Computing the activation selection for activation_selection_contrastive_for_toxic_freq_AB")
    indice = len(prompts)

    shuffled_indices = list(range(indice))
    random.shuffle(shuffled_indices)

    ques_feature_acts_list = []
    pos_feature_acts_list = []
    neg_feature_acts_list = []

    for i in tqdm(shuffled_indices, desc=f"act for {desc}"):
        # try:
            ques = prompts[i]["ques"]
            pos_prompt = prompts[i]["pos"]
            neg_prompt = prompts[i]["neg"]

            ques_tokens = model.to_tokens(
                ques,
                padding_side="right" if "gemma" in model_name else "left",
            )

            ques_tokens_len = ques_tokens.shape[-1]

            pos_tokens = model.to_tokens(
                pos_prompt,
                padding_side="right" if "gemma" in model_name else "left",
            )
            pos_attention_mask = torch.ones_like(pos_tokens)
            pos_attention_mask[pos_tokens == model.tokenizer.pad_token_id] = 0

            pos_cache = get_cache_and_logits_act(model, pos_tokens, pos_attention_mask, sae)

            pos_feature_acts = pos_cache["hook_sae_acts_post"]
            pos_feature_acts = pos_feature_acts * pos_attention_mask.unsqueeze(-1)

            ques_feature_acts = pos_feature_acts[:, :ques_tokens_len, :]
            pos_feature_acts = pos_feature_acts[:, ques_tokens_len:, :]
            pos_feature_acts = pos_feature_acts[:, -2:-1, :]


            ques_feature_acts = einops.reduce(
                ques_feature_acts,
                "batch ques feature_nums -> batch feature_nums",
                "mean",
            )

            pos_feature_acts = einops.reduce(
                pos_feature_acts,
                "batch pos feature_nums -> batch feature_nums",
                "mean",
            )

            neg_tokens = model.to_tokens(
                neg_prompt,
                padding_side="right" if "gemma" in model_name else "left",
            )

            neg_attention_mask = torch.ones_like(neg_tokens)
            neg_attention_mask[neg_tokens == model.tokenizer.pad_token_id] = 0

            neg_cache = get_cache_and_logits_act(model, neg_tokens, neg_attention_mask, sae)

            neg_feature_acts = neg_cache["hook_sae_acts_post"]
            neg_feature_acts = neg_feature_acts * neg_attention_mask.unsqueeze(-1)
            neg_feature_acts = neg_feature_acts[:, ques_tokens_len:, :]
            neg_feature_acts = neg_feature_acts[:, -2:-1, :]

            neg_feature_acts = einops.reduce(
                neg_feature_acts,
                "batch neg feature_nums -> batch feature_nums",
                "mean",
            )

            ques_feature_acts_list.append(ques_feature_acts)
            pos_feature_acts_list.append(pos_feature_acts)
            neg_feature_acts_list.append(neg_feature_acts)
        # except RuntimeError as e:
        #     if "out of memory" in str(e):
        #         print(f"Skipping due to OOM at index {i}")
        #         torch.cuda.empty_cache()
        #     else:
        #         raise e
    ques_feature_acts = torch.cat(ques_feature_acts_list, dim=0)
    pos_feature_acts = torch.cat(pos_feature_acts_list, dim=0)
    neg_feature_acts = torch.cat(neg_feature_acts_list, dim=0)
    feature_score = (
        pos_feature_acts.mean(0) - neg_feature_acts.mean(0)
    )

    pos_feature_freq = (pos_feature_acts > 0).float().sum(0)
    print(pos_feature_freq.shape)
    print(pos_feature_freq)
    neg_feature_freq = (neg_feature_acts > 0).float().sum(0)
    print(neg_feature_freq.shape)
    print(neg_feature_freq)

    pos_act_mean = pos_feature_acts.mean(0)
    print(pos_act_mean.shape)
    print(pos_act_mean)
    neg_act_mean = neg_feature_acts.mean(0)
    print(neg_act_mean.shape)
    print(neg_act_mean)

    return feature_score, pos_feature_freq, neg_feature_freq, pos_act_mean, neg_act_mean

def activation_selection_contrastive_for_only_sys(
    model, 
    sae, 
    prompts,
    neg_prompts=None,
    prefix_prompts=None, 
    batch_size=4, 
    model_name=None, 
    desc="None",
    mean=True,
    model_name_or_path="None",
):
    print(f"Computing the activation selection for activation_selection_contrastive_for_prompt")
    indice = len(prompts)

    shuffled_indices = list(range(indice))
    random.shuffle(shuffled_indices)

    ques_chosen_feature_acts_list = []
    ques_rej_feature_acts_list = []
    pos_feature_acts_list = []
    neg_feature_acts_list = []

    for i in tqdm(shuffled_indices, desc=f"act for {desc}"):
        # try:
            ques_chosen = prompts[i]["ques_chosen"]
            ques_rej = prompts[i]["ques_rej"]
            # if "llama-3.1-8b-instruct" in model_name_or_path.lower():
            #     ques = llama3_chat_input_format(ques, model_output=None, system_prompt=None)
            pos_prompt = prompts[i]["pos"]
            neg_prompt = prompts[i]["neg"]

            ques_chosen_tokens = model.to_tokens(
                ques_chosen,
                padding_side="right" if "gemma" in model_name else "left",
            )
            ques_chosen_tokens_len = ques_chosen_tokens.shape[-1]
            
            ques_rej_tokens = model.to_tokens(
                ques_rej,
                padding_side="right" if "gemma" in model_name else "left",
            )
            ques_rej_tokens_len = ques_rej_tokens.shape[-1]

            pos_tokens = model.to_tokens(
                pos_prompt,
                padding_side="right" if "gemma" in model_name else "left",
            )
            pos_attention_mask = torch.ones_like(pos_tokens)
            pos_attention_mask[pos_tokens == model.tokenizer.pad_token_id] = 0

            print("######p_activations#######")
            pos_cache = get_cache_and_logits_act(model, pos_tokens, pos_attention_mask, sae)
            # torch.save(
            #     pos_cache["activations"],
            #     "./data/safety/prompt_vector/only_sys_self_test/sae_caa_vector_it_old_bf16/p_activations.pt"
            # )
            pos_feature_acts = pos_cache["hook_sae_acts_post"]
            pos_feature_acts = pos_feature_acts * pos_attention_mask.unsqueeze(-1)

            ques_chosen_feature_acts = pos_feature_acts[:, :ques_chosen_tokens_len, :]
            pos_feature_acts = pos_feature_acts[:, ques_chosen_tokens_len:, :]

            ques_chosen_feature_acts = einops.reduce(
                ques_chosen_feature_acts,
                "batch ques feature_nums -> batch feature_nums",
                "mean",
            )

            pos_feature_acts = einops.reduce(
                pos_feature_acts,
                "batch pos feature_nums -> batch feature_nums",
                "mean",
            )

            neg_tokens = model.to_tokens(
                neg_prompt,
                padding_side="right" if "gemma" in model_name else "left",
            )
            neg_attention_mask = torch.ones_like(neg_tokens)
            neg_attention_mask[neg_tokens == model.tokenizer.pad_token_id] = 0

            print("######n_activations#######")
            neg_cache = get_cache_and_logits_act(model, neg_tokens, neg_attention_mask, sae)
            # torch.save(
            #     neg_cache["activations"],
            #     "./data/safety/prompt_vector/only_sys_self_test/sae_caa_vector_it_old_bf16/n_activations.pt"
            # )
            neg_feature_acts = neg_cache["hook_sae_acts_post"]
            neg_feature_acts = neg_feature_acts * neg_attention_mask.unsqueeze(-1)

            ques_rej_feature_acts = neg_feature_acts[:, :ques_rej_tokens_len, :]
            neg_feature_acts = neg_feature_acts[:, ques_rej_tokens_len:, :]
            
            ques_rej_feature_acts = einops.reduce(
                ques_rej_feature_acts,
                "batch ques feature_nums -> batch feature_nums",
                "mean",
            )

            neg_feature_acts = einops.reduce(
                neg_feature_acts,
                "batch neg feature_nums -> batch feature_nums",
                "mean",
            )

            ques_chosen_feature_acts_list.append(ques_chosen_feature_acts)
            ques_rej_feature_acts_list.append(ques_rej_feature_acts)
            pos_feature_acts_list.append(pos_feature_acts)
            neg_feature_acts_list.append(neg_feature_acts)
        # except RuntimeError as e:
        #     if "out of memory" in str(e):
        #         print(f"Skipping due to OOM at index {i}")
        #         torch.cuda.empty_cache()
        #     else:
        #         raise e
    ques_chosen_feature_acts = torch.cat(ques_chosen_feature_acts_list, dim=0)
    ques_rej_feature_acts = torch.cat(ques_rej_feature_acts_list, dim=0)
    pos_feature_acts = torch.cat(pos_feature_acts_list, dim=0)
    neg_feature_acts = torch.cat(neg_feature_acts_list, dim=0)
    # pdb.set_trace()
    feature_score = (
        pos_feature_acts.mean(0) - neg_feature_acts.mean(0)
    )
    print("feature_score: ", feature_score.shape)
    print("feature_score: ", feature_score)
    pos_feature_freq = (pos_feature_acts > 0).float().sum(0)
    print("pos_feature_freq: ", pos_feature_freq.shape)
    print("pos_feature_freq: ", pos_feature_freq)
    neg_feature_freq = (neg_feature_acts > 0).float().sum(0)
    print("neg_feature_freq: ", neg_feature_freq.shape)
    print("neg_feature_freq: ", neg_feature_freq)

    pos_act_mean = pos_feature_acts.mean(0)
    print("pos_act_mean: ", pos_act_mean.shape)
    print("pos_act_mean: ", pos_act_mean)
    neg_act_mean = neg_feature_acts.mean(0)
    print("neg_act_mean: ", neg_act_mean.shape)
    print("neg_act_mean: ", neg_act_mean)

    return feature_score, pos_feature_freq, neg_feature_freq, pos_act_mean, neg_act_mean

def load_gemma_2_sae(
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
    print(f"Loading Gemma 2 SAE with config: {sae_cfg}")
    sae = SAE(sae_cfg)
    sae.load_state_dict(state_dict)

    # No sparsity tensor for Gemma 2 SAEs
    log_sparsity = None

    return sae, log_sparsity


if __name__ == "__main__":

    sae_path = "/mnt/20t/msy/models/gemma-scope-9b-it-res/layer_20/width_131k/average_l0_24"
    sae, _ = load_gemma_2_sae(sae_path=sae_path, device="cuda:5")
    llama_sae, _ = load_sae_from_dir("/mnt/20t/msy/shae/exp/llama-3.1-jumprelu-resid_post/layer_20/ef16")
    import pdb;pdb.set_trace()
