import os
import pandas as pd
from datetime import datetime
import statistics
from numpy import *
import random
import json
from hashlib import md5
from tqdm import tqdm, trange
import os.path
import numpy as np
import sys
import torch
import argparse
import scipy
import nltk
import glob

def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)

def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)

def rewrite_json(path, data):
    with open(path, 'a') as file:
    # 使用json.dump将数据写入JSON文件，ensure_ascii参数设置为False
        json.dump(data, file, indent=4)
        file.write(',')
        file.write('\n')
        file.flush()

def eval(data, batch_size=32):
    ngram = []
    for i in range(0, len(data), batch_size):
        if i + batch_size > len(data):
            batch = data[i:len(data)]
        else:
            batch = data[i:i + batch_size]
        n_gram_temp = [n_gram_entropy([text,]) for text in batch]  #n_gram_entropy() return float value
        ngram += n_gram_temp
    fluency = mean(ngram)
    return fluency


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data      


def write_json(path, data, case_id = None, data_all = None):
    if data_all is None:
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)
    else:
        with open(path, 'a') as file:
            if case_id == 0:
                file.write("[")
            json.dump(data, file, indent=4)
            if case_id == data_all-1:
                file.write('\n')
                file.write("]")
            else:
                file.write(',')
                file.write('\n')
                file.flush()


if __name__ == '__main__':

    # data_files = glob.glob('/mnt/16t/xzwnlp/SaeEdit/shae/data/toxic_DINM/results/baseline/gemma*.json')

    data_files = [
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num2/gemma-2-9b_results_safety/baseline/eval_realtoxicity_test500/gemma-2-9b_steerDINMnum2_caa__layer24_multiplier1.0.result.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num4/gemma-2-9b_results_safety/baseline/eval_realtoxicity_test500/gemma-2-9b_steerDINMnum4_caa__layer24_multiplier1.0.result.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num8/gemma-2-9b_results_safety/baseline/eval_realtoxicity_test500/gemma-2-9b_steerDINMnum8_caa__layer24_multiplier1.0.result.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num16/gemma-2-9b_results_safety/baseline/eval_realtoxicity_test500/gemma-2-9b_steerDINMnum16_caa__layer24_multiplier1.0.result.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num32/gemma-2-9b_results_safety/baseline/eval_realtoxicity_test500/gemma-2-9b_steerDINMnum32_caa__layer24_multiplier1.0.result.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num64/gemma-2-9b_results_safety/baseline/eval_realtoxicity_test500/gemma-2-9b_steerDINMnum64_caa__layer24_multiplier1.0.result.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num128/gemma-2-9b_results_safety/baseline/eval_realtoxicity_test500/gemma-2-9b_steerDINMnum128_caa__layer24_multiplier1.0.result.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num256/gemma-2-9b_results_safety/baseline/eval_realtoxicity_test500/gemma-2-9b_steerDINMnum256_caa__layer24_multiplier1.0.result.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num512/gemma-2-9b_results_safety/baseline/eval_realtoxicity_test500/gemma-2-9b_steerDINMnum512_caa__layer24_multiplier1.0.result.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num1024/gemma-2-9b_results_safety/baseline/eval_realtoxicity_test500/gemma-2-9b_steerDINMnum1024_caa__layer24_multiplier1.0.result.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num2048/gemma-2-9b_results_safety/baseline/eval_realtoxicity_test500/gemma-2-9b_steerDINMnum2048_caa__layer24_multiplier1.0.result.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num2/gemma-2-9b_results_safety/baseline/sae_caa_vector_act_and_fre_trim/eval_realtoxicity_test500/trim0.35/gemma-2-9b_steerDINMnum2_sae_caa_layer24_ef16_resid_post_act_and_fre_trim_adpcoeff.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num4/gemma-2-9b_results_safety/baseline/sae_caa_vector_act_and_fre_trim/eval_realtoxicity_test500/trim0.35/gemma-2-9b_steerDINMnum4_sae_caa_layer24_ef16_resid_post_act_and_fre_trim_adpcoeff.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num8/gemma-2-9b_results_safety/baseline/sae_caa_vector_act_and_fre_trim/eval_realtoxicity_test500/trim0.35/gemma-2-9b_steerDINMnum8_sae_caa_layer24_ef16_resid_post_act_and_fre_trim_adpcoeff.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num16/gemma-2-9b_results_safety/baseline/sae_caa_vector_act_and_fre_trim/eval_realtoxicity_test500/trim0.35/gemma-2-9b_steerDINMnum16_sae_caa_layer24_ef16_resid_post_act_and_fre_trim_adpcoeff.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num32/gemma-2-9b_results_safety/baseline/sae_caa_vector_act_and_fre_trim/eval_realtoxicity_test500/trim0.35/gemma-2-9b_steerDINMnum32_sae_caa_layer24_ef16_resid_post_act_and_fre_trim_adpcoeff.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num64/gemma-2-9b_results_safety/baseline/sae_caa_vector_act_and_fre_trim/eval_realtoxicity_test500/trim0.35/gemma-2-9b_steerDINMnum64_sae_caa_layer24_ef16_resid_post_act_and_fre_trim_adpcoeff.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num128/gemma-2-9b_results_safety/baseline/sae_caa_vector_act_and_fre_trim/eval_realtoxicity_test500/trim0.35/gemma-2-9b_steerDINMnum128_sae_caa_layer24_ef16_resid_post_act_and_fre_trim_adpcoeff.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num256/gemma-2-9b_results_safety/baseline/sae_caa_vector_act_and_fre_trim/eval_realtoxicity_test500/trim0.35/gemma-2-9b_steerDINMnum256_sae_caa_layer24_ef16_resid_post_act_and_fre_trim_adpcoeff.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num512/gemma-2-9b_results_safety/baseline/sae_caa_vector_act_and_fre_trim/eval_realtoxicity_test500/trim0.35/gemma-2-9b_steerDINMnum512_sae_caa_layer24_ef16_resid_post_act_and_fre_trim_adpcoeff.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num1024/gemma-2-9b_results_safety/baseline/sae_caa_vector_act_and_fre_trim/eval_realtoxicity_test500/trim0.35/gemma-2-9b_steerDINMnum1024_sae_caa_layer24_ef16_resid_post_act_and_fre_trim_adpcoeff.json",
        "./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/num2048/gemma-2-9b_results_safety/baseline/sae_caa_vector_act_and_fre_trim/eval_realtoxicity_test500/trim0.35/gemma-2-9b_steerDINMnum2048_sae_caa_layer24_ef16_resid_post_act_and_fre_trim_adpcoeff.json",
    ]
    
    for data_file in data_files:
        data_name = os.path.basename(data_file)
        output_dir = data_file
        if not os.path.exists(output_dir):
            continue

        result_init = []
        result = read_json(output_dir)
        for data in result:
            pred_value = data.get("pred", None)
            if pred_value is not None:
                result_init.append(pred_value)
        fluency = eval(result_init, batch_size=64)
        final_result = {
            "name": data_name,
            "fluency": fluency,
        }
        final_dir = os.path.dirname(output_dir)
        # rewrite_json(f'{final_dir}/final_result_clean_eos.json', final_result)
        rewrite_json(f'./data/safety/toxic_DINM_pt_data_size/toxic_DINM_pt_data_size_eval_real/final_fluency_result_wo_clean.json', final_result)
        print(f'{data_name} is all done')



