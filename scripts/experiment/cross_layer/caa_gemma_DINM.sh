device=0
mode=safety

model_name=gemma-2-9b
model_name_or_path=/data2/xzwnlp/model/gemma-2-9b
data_names=(toxic_DINM_pt)
layers=(20 21 22 23 24 25 26 27)
layer_num=${#layers[@]}

max_new_tokens=(50)
max_new_tokens=${max_new_tokens[0]}

for data_name in ${data_names[@]}; do
    for eval_data_name in toxic_DINM_test500 realtoxicity_test500; do
        for ((i=0; i<${layer_num}; i++)); do
            layer=${layers[$i]}
            log_path=./results/${data_name}/${model_name}_results_${mode}/logs/cross_layer/caa/eval_${eval_data_name}/${model_name}_steer${data_name}_caa__layer${layer}.result.log
            file_name=$(basename ${log_path})

            log_dir=$(dirname ${log_path})
            if [ ! -d "${log_dir}" ]; then
                mkdir -p "${log_dir}"
            fi

            output_file=./results/${data_name}/${model_name}_results_${mode}/cross_layer/caa/eval_${eval_data_name}/${model_name}_steer${data_name}_caa__layer${layer}.result.json

            CUDA_VISIBLE_DEVICES=${device} python ./baseline/steering_caa.py \
                --mode ${mode} \
                --layers ${layer} \
                --max_new_tokens ${max_new_tokens} \
                --multipliers 1 \
                --eval_data_name ${eval_data_name} \
                --model_name ${model_name} \
                --data_name ${data_name} \
                --data_path ./data/safety \
                --model_name_or_path ${model_name_or_path} \
                --output_file ${output_file} > ${log_path} 2>&1 

        done
    done
done
