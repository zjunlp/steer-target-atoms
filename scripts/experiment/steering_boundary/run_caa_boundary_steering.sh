device=0
mode=safety

model_name=gemma-2-9b-it
model_name_or_path=/disk3/wmr/hugging_cache/gemma-2-9b-it
data_names=(
    toxic_DINM_it
)
layers=(20)
multipliers=(-10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10)

layer_num=${#layers[@]}
multiplier_num=${#multipliers[@]}

max_new_tokens=(50)
max_new_tokens=${max_new_tokens[0]}

data_path=./data/safety

for data_name in ${data_names[@]}; do
    for eval_data_name in toxic_DINM_test500; do
        for ((i=0; i<${layer_num}; i++)); do
            layer=${layers[$i]}
            for ((j=0; j<${multiplier_num}; j++)); do
                multiplier=${multipliers[$j]}
                log_path=./results/${data_name}/${model_name}_results_${mode}/logs/steering_boundary/caa_for_bounds/eval_${eval_data_name}/${model_name}_steer${data_name}_caa_layer${layer}_multiplier${multiplier}.result.log

                log_dir=$(dirname ${log_path})
                if [ ! -d "${log_dir}" ]; then
                    mkdir -p "${log_dir}"
                fi

                output_file=./results/${data_name}/${model_name}_results_${mode}/steering_boundary/caa_for_bounds/eval_${eval_data_name}/${model_name}_steer${data_name}_caa_layer${layer}.result.json

                CUDA_VISIBLE_DEVICES=${device} python ./baseline/steering_caa.py \
                    --mode ${mode} \
                    --layers ${layer} \
                    --max_new_tokens ${max_new_tokens} \
                    --multipliers ${multiplier} \
                    --eval_data_name ${eval_data_name} \
                    --model_name ${model_name} \
                    --data_name ${data_name} \
                    --data_path ${data_path} \
                    --model_name_or_path ${model_name_or_path} \
                    --output_file ${output_file} > ${log_path} 2>&1 

            done
        done
    done
done