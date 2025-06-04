pred_file_list=(
    ./results/toxic_DINM_pt_data_size/gemma-2-9b_results_safety/caa/eval_realtoxicity_test500
    ./results/toxic_DINM_pt_data_size/gemma-2-9b_results_safety/sta/eval_realtoxicity_test500
)

for pred_file in ${pred_file_list[@]}; do
    python ./evaluate_safety/eval_realtoxicity_api.py \
        --pred_file ${pred_file}
done
