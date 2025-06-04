device=0
mode=safety

model_name=gemma-2-9b-it
model_name_or_path=/disk3/wmr/hugging_cache/gemma-2-9b-it
data_names=(
    only_sys_ax
    only_sys_self
)
layers=(20)
multipliers=(0 1 -1 2 -2 3 -3)
layer_num=${#layers[@]}
multiplier_num=${#multipliers[@]}

for data_name in ${data_names[@]}; do
    for eval_data_name in toxic_DINM_test500; do
        for ((i=0; i<${layer_num}; i++)); do
            layer=${layers[$i]}
            for ((j=0; j<${multiplier_num}; j++)); do
                multiplier=${multipliers[$j]}
                log_path=./results/prompt_vector/${data_name}/${model_name}_results_${mode}/logs/caa/eval_${eval_data_name}/${model_name}_steer${data_name}_caa__layer${layer}_multiplier${multiplier}.result.log

                log_dir=$(dirname ${log_path})
                if [ ! -d "${log_dir}" ]; then
                    mkdir -p "${log_dir}"
                fi

                output_file=./results/prompt_vector/${data_name}/${model_name}_results_${mode}/caa/eval_${eval_data_name}/${model_name}_steer${data_name}_caa__layer${layer}.result.json

                CUDA_VISIBLE_DEVICES=${device} python ./baseline/steering_caa.py \
                    --mode ${mode} \
                    --layers ${layer} \
                    --max_new_tokens 50 \
                    --multipliers ${multiplier} \
                    --eval_data_name ${eval_data_name} \
                    --model_name ${model_name} \
                    --data_name ${data_name} \
                    --data_path ./data/safety/prompt_vector \
                    --model_name_or_path ${model_name_or_path} \
                    --output_file ${output_file} > ${log_path} 2>&1 

            done
        done
    done
done
