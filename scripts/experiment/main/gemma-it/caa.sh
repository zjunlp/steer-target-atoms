device=0
mode=safety

model_name=gemma-2-9b-it
model_name_or_path=/disk3/wmr/hugging_cache/gemma-2-9b-it
data_names=(toxic_DINM_it)
layers=(20)
layer_num=${#layers[@]}

max_new_tokens=(50)
max_new_tokens=${max_new_tokens[0]}

data_path=./data/safety

for data_name in ${data_names[@]}; do
    for eval_data_name in toxic_DINM_it realtoxicity gsm; do
        for ((i=0; i<${layer_num}; i++)); do
            layer=${layers[$i]}
            log_path=./results/${data_name}/${model_name}_results_${mode}/logs/main/caa/eval_${eval_data_name}/${model_name}_steer${data_name}_caa__layer${layer}.result.log

            log_dir=$(dirname ${log_path})
            if [ ! -d "${log_dir}" ]; then
                mkdir -p "${log_dir}"
            fi

            output_file=./results/${data_name}/${model_name}_results_${mode}/main/caa/eval_${eval_data_name}/${model_name}_steer${data_name}_caa__layer${layer}.result.json

            CUDA_VISIBLE_DEVICES=${device} python ./baseline/steering_caa.py \
                --mode ${mode} \
                --layers ${layer} \
                --max_new_tokens ${max_new_tokens} \
                --multipliers 1 \
                --eval_data_name ${eval_data_name} \
                --model_name ${model_name} \
                --data_name ${data_name} \
                --data_path ${data_path} \
                --model_name_or_path ${model_name_or_path} \
                --output_file ${output_file} > ${log_path} 2>&1 

        done
    done
done


eval_data_name=mmlu

layer_num=${#layers[@]}
for data_name in ${data_names[@]}; do
    caa_vector_root=${data_path}/${data_name}/caa_vector_it/${model_name}_${mode}
    for ((i=0; i<${layer_num}; i++)); do
        layer=${layers[$i]}
        output_file=./results/${data_name}/${model_name}_results_${mode}/main/caa/eval_${eval_data_name}_qa/${model_name}_steer${data_name}_caa_layer${layer}.result.json
        log_path=./results/${data_name}/${model_name}_results_${mode}/logs/main/caa/eval_${eval_data_name}_qa/${model_name}_steer${data_name}_caa_layer${layer}.result.log

        # Check if the directory exists, if not, create it
        log_dir=$(dirname ${log_path})
        if [ ! -d "${log_dir}" ]; then
            mkdir -p "${log_dir}"
        fi

        CUDA_VISIBLE_DEVICES=${device} python ./baseline/caa_safety_mmlu.py \
            --mode ${mode} \
            --layers ${layer} \
            --qa \
            --multipliers 1 \
            --model_name ${model_name} \
            --data_path ${data_path} \
            --data_name ${data_name} \
            --eval_data_name ${eval_data_name} \
            --caa_vector_root ${caa_vector_root} \
            --model_name_or_path ${model_name_or_path} \
            --output_file ${output_file} > ${log_path} 2>&1

    done
done
