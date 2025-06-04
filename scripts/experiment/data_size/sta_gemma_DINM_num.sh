device=0
vector_type=act_and_fre_trim
mode=safety

data_names=(
    num2048
    num1024
    num512
    num256
    num128
    num64
    num32
    num16
    num8
    num4
    num2
)
layers=(24)
mymultis=(1)
trims=(0.35)
model_name=gemma-2-9b
model_name_or_path=/data2/xzwnlp/model/gemma-2-9b
data_path=./data/safety/toxic_DINM_pt_data_size
max_new_tokens=(50)
max_new_tokens=${max_new_tokens[0]}
layer_num=${#layers[@]}
trim_num=${#trims[@]}
hook_module=resid_post

for data_name in ${data_names[@]}; do
    vector_root=${data_path}/${data_name}/sae_caa_vector_pt/${model_name}_${mode}/act_and_fre_trim/steering_vector
    caa_vector_root=${data_path}/${data_name}/caa_vector_pt/${model_name}_${mode}

    for eval_data_name in toxic_DINM_test500 realtoxicity_test500; do
        for ((i=0; i<${layer_num}; i++)); do
            layer=${layers[$i]}
            for ((j=0; j<${trim_num}; j++)); do
                trim=${trims[$j]}
                output_file=./results/toxic_DINM_pt_data_size/${model_name}_results_${mode}/sta/eval_${eval_data_name}/trim${trim}/${model_name}_steerDINM${data_name}_sae_caa_layer${layer}_ef16_${hook_module}_${vector_type}.result.json
                log_path=./results/toxic_DINM_pt_data_size/${model_name}_results_${mode}/logs/sta/eval_${eval_data_name}/trim${trim}/${model_name}_steerDINM${data_name}_sae_caa_layer${layer}_ef16_${hook_module}_${vector_type}.result.log

                # Check if the directory exists, if not, create it
                log_dir=$(dirname ${log_path})
                if [ ! -d "${log_dir}" ]; then
                    mkdir -p "${log_dir}"
                fi

                CUDA_VISIBLE_DEVICES=${device} python ./baseline/our_sae_caa_safety.py \
                    --max_new_tokens ${max_new_tokens} \
                    --mymultis ${mymultis} \
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
