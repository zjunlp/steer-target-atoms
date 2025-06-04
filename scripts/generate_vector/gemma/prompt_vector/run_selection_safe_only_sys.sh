device=0

model_name_or_path=/disk3/wmr/hugging_cache/gemma-2-9b-it
data_dir=./data/safety/prompt_vector
model_name=gemma-2-9b-it

sae_paths=(
    /disk3/wmr/hugging_cache/gemma-scope-9b-it-res/layer_20/width_16k/average_l0_91
)

suffix=16k

hook_module=resid_post
select_type=only_sys
mode=safety
sae_num=${#sae_paths[@]}
# for data_name in only_sys_self; do
for data_name in only_sys_self only_sys_ax; do
    for ((i=0; i<${sae_num}; i++)); do

        sae_path=${sae_paths[$i]}

        if [[ ${sae_path} =~ layer_([0-9]+) ]]; then
            layer=${BASH_REMATCH[1]}
            echo "Layer: $layer, Hook Module: $hook_module"
        else
            echo "Error: 'layer' not found in sae_path: ${sae_path}"
            exit 1
        fi

        output_file=${data_dir}/${data_name}/sae_caa_vector_it/${model_name}_${mode}/sae_vector/feature_${select_type}_${model_name}_layer${layer}_${suffix}_${hook_module}.json
        log_path=${data_dir}/${data_name}/sae_caa_vector_it/${model_name}_${mode}/sae_vector/logs/feature_${select_type}_${model_name}_layer${layer}_${suffix}_${hook_module}.log

        # Check if the directory exists, if not, create it
        output_dir=$(dirname ${output_file})
        if [ ! -d "${output_dir}" ]; then
            mkdir -p "${output_dir}"
        fi

        log_dir=$(dirname ${log_path})
        if [ ! -d "${log_dir}" ]; then
            mkdir -p "${log_dir}"
        fi

        CUDA_VISIBLE_DEVICES=${device} python sae_feature_selection.py \
            --mode ${mode} \
            --model_name ${model_name} \
            --model_name_or_path ${model_name_or_path} \
            --sae_path ${sae_path} \
            --data_file ${data_dir}/${data_name}/train.csv \
            --data_name ${data_name} \
            --steering_vector_name ${model_name}_sae_layer${layer}_${hook_module}_${suffix}_steering_vector.pt \
            --output_file ${output_file} \
            --select_type ${select_type} > ${log_path} 2>&1  

    done
done