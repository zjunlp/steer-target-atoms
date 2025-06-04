# caa vector generation script for gemma-2-9b and gemma-2-9b-it for DINM data
bash ./scripts/generate_vector/gemma/caa/generate_vectors_gemma_pt_DINM.sh

bash ./scripts/generate_vector/gemma/caa/generate_vectors_gemma_it_DINM.sh

bash ./scripts/generate_vector/gemma/caa/generate_vectors_gemma_pt_DINM_data_size.sh



# sta vector generation script for gemma-2-9b and gemma-2-9b-it for DINM data
bash ./scripts/generate_vector/gemma/sta/run_selection_safe_gemma_pt_DINM.sh
bash ./scripts/generate_vector/gemma/sta/run_save_gemma_pt_act-and-fre_trim_DINM.sh

bash ./scripts/generate_vector/gemma/sta/run_selection_safe_gemma_it_DINM.sh
bash ./scripts/generate_vector/gemma/sta/run_save_gemma_it_act-and-fre_trim_DINM.sh

bash ./scripts/generate_vector/gemma/sta/run_selection_safe_gemma_pt_DINM_data_size.sh
bash ./scripts/generate_vector/gemma/sta/run_save_gemma_pt_act-and-fre_trim_DINM_data_size.sh

# prompt vector generation script for gemma-2-9b-it for safety data
bash ./scripts/generate_vector/gemma/prompt_vector/generate_vectors_gemma_sys.sh

bash ./scripts/generate_vector/gemma/prompt_vector/run_selection_safe_only_sys.sh

bash ./scripts/generate_vector/gemma/prompt_vector/run_save_gemma_it_act-and-fre_trim_prompt_vector.sh