device=0

mode=safety

data_path=./data/safety
model_name=gemma-2-9b
data_names=(
    toxic_DINM_pt
)
model_name_or_path=./model/gemma-2-9b # replace ./model/gemma-2-9b with your own model path

for eval_data_name in ${data_names[@]}; do
    data_name=${eval_data_name}
    log_path=${data_path}/${data_name}/caa_vector_pt/${model_name}_${mode}/logs/${model_name}_${data_name}.log
    echo "Starting the script..."
     # 创建日志目录
    log_dir=$(dirname ${log_path})
    echo "Log directory: ${log_dir}"
    if [ ! -d "${log_dir}" ]; then
        echo "Creating log directory..."
        mkdir -p "${log_dir}"
    else
        echo "Log directory already exists."
    fi

    # 打印生成的日志路径
    echo "Log path: ${log_path}"

    echo "Running Python script for ${data_name}..."
    # --layers $(seq 0 41) \

    CUDA_VISIBLE_DEVICES=${device} python ./baseline/generate_vectors.py \
        --mode ${mode} \
        --layers $(seq 0 41) \
        --save_activations \
        --model_name ${model_name} \
        --data_path ${data_path} \
        --data_name ${data_name} \
        --data_type safety \
        --model_name_or_path ${model_name_or_path} > ${log_path} 2>&1 

done
echo "Script execution completed."
