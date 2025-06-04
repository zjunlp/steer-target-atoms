device=0
mode=safety

model_name=gemma-2-9b
model_name_or_paths=(/data2/xzwnlp/model/gemma-2-9b)

model_num=${#model_name_or_paths[@]}

data_names=(
    toxic_DINM_pt
)
data_path=./data/safety

data_name=toxic_DINM_pt
data_path=./data/safety
model_num=${#model_name_or_paths[@]}
for eval_data_name in toxic_DINM_pt realtoxicity gsm; do
    for ((i=0; i<${model_num}; i++)); do

        model_name_or_path=${model_name_or_paths[$i]}
        log_path=./results/${data_name}/${model_name}_results_${mode}/logs/main/base/eval_${eval_data_name}/${model_name}_base_${eval_data_name}.result.log

        # Check if the directory exists, if not, create it
        log_dir=$(dirname ${log_path})
        if [ ! -d "${log_dir}" ]; then
            mkdir -p "${log_dir}"
        fi


        output_file=./results/${data_name}/${model_name}_results_${mode}/main/base/eval_${eval_data_name}/${model_name}_base_${eval_data_name}.result.json
        file_name=$(basename ${output_file})


        CUDA_VISIBLE_DEVICES=${device} python ./baseline/steering_base.py \
            --model_name ${model_name} \
            --data_name ${data_name} \
            --max_new_tokens 50 \
            --eval_data_name ${eval_data_name} \
            --data_path ${data_path} \
            --model_name_or_path ${model_name_or_path} \
            --output_file ${output_file} > ${log_path} 2>&1 

    done
done

eval_data_name=mmlu

for data_name in ${data_names[@]}; do
    for ((i=0; i<${model_num}; i++)); do

        model_name_or_path=${model_name_or_paths[$i]}
        train_mode=${train_modes[$i]}

        output_file=./results/${data_name}/${model_name}_results_${mode}/main/base/eval_${eval_data_name}/${model_name}_base_${data_name}_base.result.json
        log_path=./results/${data_name}/${model_name}_results_${mode}/logs/main/base/eval_${eval_data_name}/${model_name}_base_${data_name}_base.result.log

        # Check if the directory exists, if not, create it
        log_dir=$(dirname ${log_path})
        if [ ! -d "${log_dir}" ]; then
            mkdir -p "${log_dir}"
        fi

        CUDA_VISIBLE_DEVICES=${device} python ./baseline/base_safety_mmlu.py \
            --mode ${mode} \
            --model_name ${model_name} \
            --data_path ${data_path} \
            --data_name ${data_name} \
            --eval_data_name ${eval_data_name} \
            --model_name_or_path ${model_name_or_path} \
            --output_file ${output_file} > ${log_path} 2>&1

    done
done