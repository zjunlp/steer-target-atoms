device=0
vector_type=act_and_fre_trim
mode=safety

data_name=toxic_DINM_it
layers=(20)
mymultis=(-10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10)
trims=(0.35)
model_name=gemma-2-9b-it
model_name_or_path=/disk3/wmr/hugging_cache/gemma-2-9b-it
data_path=./data/safety
vector_root=${data_path}/${data_name}/sae_caa_vector_it/${model_name}_${mode}/act_and_fre_trim/steering_vector
caa_vector_root=${data_path}/${data_name}/caa_vector_it/${model_name}_${mode}
hook_module=resid_post
max_new_tokens=(50)
max_new_tokens=${max_new_tokens[0]}

layer_num=${#layers[@]}
trim_num=${#trims[@]}
mymulti_num=${#mymultis[@]}
for eval_data_name in toxic_DINM_test500; do
    for ((i=0; i<${layer_num}; i++)); do
        layer=${layers[$i]}
        for ((j=0; j<${trim_num}; j++)); do
            trim=${trims[$j]}
            for ((g=0; g<${mymulti_num}; g++)); do
                mymulti=${mymultis[$g]}
                output_file=./results/${data_name}/${model_name}_results_${mode}/steering_boundary/sta_for_bounds/eval_${eval_data_name}/trim${trim}/${model_name}_steer${eval_data_name}_sae_caa_layer${layer}_ef16_${hook_module}_${vector_type}_trim${trim}.result.json
                log_path=./results/${data_name}/${model_name}_results_${mode}/logs/steering_boundary/sta_for_bounds/eval_${eval_data_name}/trim${trim}/${model_name}_steer${eval_data_name}_sae_caa_layer${layer}_ef16_${hook_module}_${vector_type}_trim${trim}_mymulti${mymulti}.result.log

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
