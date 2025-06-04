device=0
mode=safety
sae_path=/disk3/wmr/hugging_cache/gemma-scope-9b-it-res/layer_20/width_16k/average_l0_91
path_dir=./data/safety/prompt_vector
model_name=gemma-2-9b-it
select_type=act_and_fre_trim
hook_module=resid_post
# layers=(20 21 22 23 24 25 26 27)
layers=(20)
trims=(0.006)

layer_num=${#layers[@]}
for data_name in only_sys_self only_sys_ax; do
    for ((i=0; i<${layer_num}; i++)); do
        layer=${layers[$i]}
        trims_num=${#trims[@]}
        sae_path_layer=${sae_path}
        echo "Layer: $layer, sae_path_layer: $sae_path_layer"
        for ((j=0; j<${trims_num}; j++)); do
            trim=${trims[$j]}
            log_path=${path_dir}/${data_name}/sae_caa_vector_it/${model_name}_${mode}/act_and_fre_trim/logs/${model_name}_${suffix}_steering_vector_sae_layer${layer}_act_and_fre_trim${trim}.log
            log_dir=$(dirname ${log_path})
            if [ ! -d "${log_dir}" ]; then
                mkdir -p "${log_dir}"
            fi
            CUDA_VISIBLE_DEVICES=${device} python ./generate_sae_caa_vector.py \
                --path_dir ${path_dir} \
                --sae_path ${sae_path_layer} \
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
