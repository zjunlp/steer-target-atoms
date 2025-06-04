pred_file_list=(
    ./results/toxic_DINM_pt/gemma-2-9b_results_safety/main/base/eval_gsm
)
# you can replace it with your own pred_file_list 

for pred_file in ${pred_file_list[@]}; do
    python ./evaluate_safety/eval_gemma_gsm.py \
        --pred_file ${pred_file}
done
