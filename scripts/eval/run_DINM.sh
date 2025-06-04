pred_file_list=(
    ./results/toxic_DINM_pt_data_size/gemma-2-9b_results_safety/caa/eval_toxic_DINM_test500
    ./results/toxic_DINM_pt_data_size/gemma-2-9b_results_safety/sta/eval_toxic_DINM_test500/trim0.35
    ./results/toxic_DINM_pt/gemma-2-9b_results_safety/main/base/eval_toxic_DINM_pt/gemma-2-9b_base_toxic_DINM_pt.result.json
)
safety_classifier_dir=/data2/xzwnlp/hugging_cache/plus_safety_classifier_all_layers

for pred_file in ${pred_file_list[@]}; do
    python ./evaluate_safety/eval_DINM.py \
        --pred_file ${pred_file} \
        --safety_classifier_dir ${safety_classifier_dir}
done
