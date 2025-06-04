
<div align="center">
    
# **Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms**
    
</div>



## 🔧 Pip Installation

To get started, simply install conda:

```shell
conda create -n sta python=3.11 -y
pip install -r requirements.txt
cd ./TransformerLens
pip install -e . # 2.4.0
cd ../trl
pip install -e . # for sft dpo training
```


## 📂 Data Preparation

**Dataset and Steering Vector**

The data for STA can be downloaded [here](https://huggingface.co/datasets/mengru/data_for_STA).


**Directory Structure**

```
steer-target-atoms
└── data
    ├── mmlu
    └── safety
```

## 💻 Run

### Steering vector

#### directly download

If you download data from [here](https://huggingface.co/datasets/mengru/data_for_STA), then you can get the steering vectors used in this paper:

- steering vecotr for Gemma-2-9b-it (./data/safety/toxic_DINM_it/sae_caa_vector_it/gemma-2-9b-it_safety/act_and_fre_trim/steering_vector)

- steering vecotr for Gemma-2-9b-pt (./data/safety/toxic_DINM_pt/sae_caa_vector_pt/gemma-2-9b_safety/act_and_fre_trim/steering_vector)

  Then, you can directly go to the [Steering the behaviors of LLMs](#steering-the-behaviors-of-llms) section.

#### Generate the steering vector by yourself

You can also generate these steering vectors using the following steps by yourself:

1. Download the sae

- Download sea for Gemma-2-9b-it from [here](https://huggingface.co/google/gemma-scope-9b-it-res/tree/main/layer_20/width_16k/average_l0_91), then replace the value of sae_paths (in ./scripts/generate_vector/gemma/sta/run_selection_safe_gemma_it_DINM.sh) with your own path.

- Download sea for Gemma-2-9b-pt from [here](https://huggingface.co/google/gemma-scope-9b-pt-res/tree/main/layer_24/width_16k/average_l0_114), then replace the value of sae_paths (in ./scripts/generate_vector/gemma/sta/run_selection_safe_gemma_pt_DINM.sh) with your own path.



2. Genetate steering vector

```shell
bash run_generate_vector.sh
```

### Steering the behaviors of LLMs 

You can steering the behaviors of LLMs by **steering vector**

```shell
bash run_main_table.sh
```
> ❗️ You should replace the value of model_name_or_path in the corresponding xx.sh file with your own model path.

### Evaluation

```shell
bash run_eval.sh
```

## 🌟 Some Important Information

This repository is developed for our STA paper. We also release [EasyEdit2](https://github.com/zjunlp/EasyEdit/blob/main/README_2.md), a unified framework for controllable editing without retraining. It integrates multiple steering methods to facilitate usage and evaluation.
Unlike this repository, which depends on TransformerLens, EasyEdit2 is independent of it.

We recommend using [EasyEdit2](https://github.com/zjunlp/EasyEdit/blob/main/README_2.md) for future research and applications.

# 📖 Citation

Please cite our paper if you use **STA** in your work.

```bibtex
@misc{wang2025STA,
      title={Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms}, 
      author={Mengru Wang, Ziwen Xu, Shengyu Mao, Shumin Deng, Zhaopeng Tu, Huajun Chen, Ningyu Zhang},
      year={2025},
      eprint={2505.20322},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
