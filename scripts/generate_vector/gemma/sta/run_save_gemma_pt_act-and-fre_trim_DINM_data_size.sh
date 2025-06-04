device=0
path_dir=./data/safety/toxic_DINM_pt_data_size
sae_paths=(
    /data2/xzwnlp/gemma-scope-9b-pt-res/layer_24/width_16k/average_l0_114
)
layers=(24)
model_name=gemma-2-9b
mode=safety
select_type=act_and_fre_trim

hook_module=resid_post
trims=(0.350)


layer_num=${#layers[@]}

for data_name in num2048 num1024 num512 num256 num128 num64 num32 num16 num8 num4 num2; do
    for ((i=0; i<${layer_num}; i++)); do
        layer=${layers[$i]}
        trims_num=${#trims[@]}
        sae_path=${sae_paths[$i]}
        echo "Layer: $layer, sae_path: $sae_path"
        for ((j=0; j<${trims_num}; j++)); do
            trim=${trims[$j]}
            log_path=${path_dir}/${data_name}/sae_caa_vector_pt/${model_name}_${mode}/act_and_fre_trim/logs/${model_name}_${suffix}_steering_vector_sae_layer${layer}_act_and_fre_trim${trim}.log
            log_dir=$(dirname ${log_path})
            if [ ! -d "${log_dir}" ]; then
                mkdir -p "${log_dir}"
            fi
            CUDA_VISIBLE_DEVICES=${device} python ./generate_sae_caa_vector.py \
                --path_dir ${path_dir} \
                --sae_path ${sae_path} \
                --data_name ${data_name} \
                --model_name ${model_name} \
                --mode ${mode} \
                --select_type ${select_type} \
                --layers ${layer} \
                --hook_module ${hook_module} \
                --trim ${trim} > ${log_path} 2>&1
        done
    done
done