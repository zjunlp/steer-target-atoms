device=0

model_name_or_path=/data2/xzwnlp/model/gemma-2-9b
data_dir=./data/safety
model_name=gemma-2-9b

sae_paths=(
    /data2/xzwnlp/gemma-scope-9b-pt-res/layer_20/width_16k/average_l0_68
    /data2/xzwnlp/gemma-scope-9b-pt-res/layer_21/width_16k/average_l0_129
    /data2/xzwnlp/gemma-scope-9b-pt-res/layer_22/width_16k/average_l0_123
    /data2/xzwnlp/gemma-scope-9b-pt-res/layer_23/width_16k/average_l0_120
    /data2/xzwnlp/gemma-scope-9b-pt-res/layer_24/width_16k/average_l0_114
    /data2/xzwnlp/gemma-scope-9b-pt-res/layer_25/width_16k/average_l0_114
    /data2/xzwnlp/gemma-scope-9b-pt-res/layer_26/width_16k/average_l0_116
    /data2/xzwnlp/gemma-scope-9b-pt-res/layer_27/width_16k/average_l0_118
)

suffix=16k

hook_module=resid_post
data_name=toxic_DINM_pt
select_type=sae_vector
mode=safety
sae_num=${#sae_paths[@]}
for ((i=0; i<${sae_num}; i++)); do

    sae_path=${sae_paths[$i]}

    if [[ ${sae_path} =~ layer_([0-9]+) ]]; then
        layer=${BASH_REMATCH[1]}
        echo "Layer: $layer, Hook Module: $hook_module"
    else
        echo "Error: 'layer' not found in sae_path: ${sae_path}"
        exit 1
    fi

    output_file=${data_dir}/${data_name}/sae_caa_vector_pt/${model_name}_${mode}/${select_type}/feature_${select_type}_${model_name}_layer${layer}_${suffix}_${hook_module}.json
    log_path=${data_dir}/${data_name}/sae_caa_vector_pt/${model_name}_${mode}/${select_type}/logs/feature_${select_type}_${model_name}_layer${layer}_${suffix}_${hook_module}.log

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
