device=0
vector_type=act_and_fre_trim
mode=safety

data_names=(
    only_sys_ax
    only_sys_self
)
layers=(20)
mymultis=(0 1 -1 2 -2 3 -3)
trims=(0.006)
model_name=gemma-2-9b-it
model_name_or_path=/disk3/wmr/hugging_cache/gemma-2-9b-it
data_path=./data/safety/prompt_vector
hook_module=resid_post
max_new_tokens=(50)
max_new_tokens=${max_new_tokens[0]}
layer_num=${#layers[@]}
trim_num=${#trims[@]}
mymulti_num=${#mymultis[@]}

for data_name in ${data_names[@]}; do
    vector_root=${data_path}/${data_name}/sae_caa_vector_it/${model_name}_${mode}/act_and_fre_trim/steering_vector
    caa_vector_root=${data_path}/${data_name}/caa_vector_it/${model_name}_${mode}
    for eval_data_name in toxic_DINM_test500; do
        for ((i=0; i<${layer_num}; i++)); do
            layer=${layers[$i]}
            for ((j=0; j<${trim_num}; j++)); do
                trim=${trims[$j]}
                for ((g=0; g<${mymulti_num}; g++)); do
                    mymulti=${mymultis[$g]}
                    output_file=./results/prompt_vector/${data_name}/${model_name}_results_${mode}/sta_${vector_type}/eval_${eval_data_name}/trim${trim}/${model_name}_steer${eval_data_name}_sae_caa_layer${layer}_ef16_${hook_module}_${vector_type}_trim${trim}.result.json
                    log_path=./results/prompt_vector/${data_name}/${model_name}_results_${mode}/logs/sta_${vector_type}/eval_${eval_data_name}/trim${trim}/${model_name}_steer${eval_data_name}_sae_caa_layer${layer}_ef16_${hook_module}_${vector_type}_trim${trim}_mymultis${mymulti}.result.log

                    # Check if the directory exists, if not, create it
                    log_dir=$(dirname ${log_path})
                    if [ ! -d "${log_dir}" ]; then
                        mkdir -p "${log_dir}"
                    fi

                    CUDA_VISIBLE_DEVICES=${device} python ./baseline/our_sae_caa_safety.py \
                        --max_new_tokens ${max_new_tokens} \
                        --mymultis ${mymulti} \
                        --vector_type ${vector_type} \
                        --mode ${mode} \
                        --layers ${layer} \
                        --model_name ${model_name} \
                        --model_name_or_path ${model_name_or_path} \
                        --data_path ${data_path}  \
                        --data_name ${data_name} \
                        --eval_data_name ${eval_data_name} \
                        --trim ${trim} \
                        --hook_module ${hook_module} \
                        --vector_root ${vector_root}\
                        --caa_vector_root ${caa_vector_root} \
                        --output_file ${output_file} > ${log_path} 2>&1

                done
            done
        done
    done
done


