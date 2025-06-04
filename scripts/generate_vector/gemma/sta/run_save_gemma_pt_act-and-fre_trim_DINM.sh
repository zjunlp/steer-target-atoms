device=0
data_name=toxic_DINM_pt
model_name=gemma-2-9b
mode=safety
select_type=act_and_fre_trim
path_dir=./data/safety
hook_module=resid_post
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
layers=(20 21 22 23 24 25 26 27)
# layers=(24)

trims=(0.350)

layer_num=${#layers[@]}

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