# caa vector generation script for gemma-2-9b and gemma-2-9b-it for DINM data (SafeEdit dataset)
bash ./scripts/generate_vector/gemma/caa/generate_vectors_gemma_pt_DINM.sh
bash ./scripts/generate_vector/gemma/caa/generate_vectors_gemma_it_DINM.sh




# sta vector generation script for gemma-2-9b and gemma-2-9b-it for DINM data (SafeEdit dataset)
bash ./scripts/generate_vector/gemma/sta/run_selection_safe_gemma_pt_DINM.sh
bash ./scripts/generate_vector/gemma/sta/run_save_gemma_pt_act-and-fre_trim_DINM.sh

bash ./scripts/generate_vector/gemma/sta/run_selection_safe_gemma_it_DINM.sh
bash ./scripts/generate_vector/gemma/sta/run_save_gemma_it_act-and-fre_trim_DINM.sh
