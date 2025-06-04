
<div align="center">
    
# **Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms**
    
</div>



## 🔧 Pip Installation

To get started, simply install conda and run:

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

The data for STA can be downloaded [here](https://huggingface.co/datasets/mengru/data_for_STA), including the steering vectors used in this paper. Alternatively, you can generate these steering vectors using run_generate_vector.sh.


**Directory Structure**

```
steer-target-atoms
└── data
    ├── mmlu
    ├── r1
    └── safety
```

## 💻 Run
```shell
bash run_main_table.sh
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
