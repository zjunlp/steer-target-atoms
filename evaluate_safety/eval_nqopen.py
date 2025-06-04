import os
from numpy import *
import json
import os.path
import torch
torch.cuda.set_device(5)

def rewrite_json(path, data):
    with open(path, 'a') as file:
    # 使用json.dump将数据写入JSON文件，ensure_ascii参数设置为False
        json.dump(data, file, indent=4)
        file.write(',')
        file.write('\n')
        file.flush()


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

def calculate_accuracy(data_file):
        with open(data_file, 'r') as file:
            data = json.load(file)
        print(len(data))
        correct_predictions = 0
        total_predictions = 0

        for item in data:
            answer = item.get('answer', [])
            pred = item.get('pred', '')
            # Check if any answer in the list is in the predicted answer
            if any([ans.lower() in pred.lower() for ans in answer]):
                correct_predictions += 1
            total_predictions += 1
        accuracy = correct_predictions / total_predictions if total_predictions else 0

        print(f"Total:{total_predictions}  Correct:{correct_predictions} Accuracy:{accuracy:.2%}")
        return accuracy

if __name__ == '__main__':

    # data_files = glob.glob('/mnt/16t/xzwnlp/SaeEdit/shae/data/toxic_DINM/results/baseline/gemma*.json')

    data_files = [
        "./data/safety/toxic_DINM_it/gemma-2-9b-it_results_safety/baseline/base/eval_nqopen-addw/gemma-2-9b-it_base_nqopen-addw.result.json",
        "./data/safety/toxic_DINM_it/gemma-2-9b-it_results_safety/baseline/eval_nqopen-addw/gemma-2-9b-it_steertoxic_DINM_it_caa__layer20_multiplier1.0.result.json",
        "./data/safety/toxic_DINM_it/gemma-2-9b-it_results_safety/sae_caa_vector_it_act_and_fre_trim/eval_nqopen-addw/trim0.35/gemma-2-9b-it_steernqopen-addw_sae_caa_layer20_ef16_resid_post_act_and_fre_trim_adpcoeff_trim0.35_mymulti_1.result.json",
        "./data/safety/toxic_DINM_it/gemma-2-9b-it_results_safety/sae_caa_vector_it_sae_vector/eval_nqopen-addw/trim0/adpcoeff/gemma-2-9b-it_steertoxic_DINM_it_sae_caa_layer20_16k_resid_post_sae_vector_adpcoeff_trim0_mymulti_1.0.result.json",
        "./data/safety/toxic_DINM_it/gemma-2-9b-it_results_safety/baseline/system_prompt_4/eval_nqopen-addw/gemma-2-9b-it_system_prompt_4_nqopen-addw.result_system_prompt.json",
        "./data/safety/toxic_DINM_it/gemma-2-9b-it_results_safety/baseline/system_prompt_6/eval_nqopen-addw/gemma-2-9b-it_system_prompt_6_nqopen-addw.result_system_prompt.json"
    ]

    # Calculate accuracy for each data file
    for data_file in data_files:
        accuracy = calculate_accuracy(data_file)
        # print(f"Accuracy for {data_file}: {accuracy * 100:.2f}%")
        final_result = {
            "name": data_file,
            "acc": accuracy * 100,
        }
        final_dir = os.path.dirname(data_file)
        # rewrite_json(f'{final_dir}/final_result_clean_eos.json', final_result)
        rewrite_json(f'{final_dir}/final_result_wo_clean.json', final_result)
        print(f'{data_file} is all done')



