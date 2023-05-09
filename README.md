<div align="center">

<p align="center">
  <img src="./assets/logo.jpg" width="300px" alt="logo">
</p>

# CodeSync: Bootstrapping Code-Text Pretrained Language Model to Detect Inconsistency Between \\ Code and Comment
__________________________

</div>

<!-- ## Table of content
- [CodeSync package](#codesync-package)
	- [Getting Started] (#getting-started)
	- [Inference] (#inference)
	- [Pre-training] (#pretrain)
		- [Installation] (#install)
	 	- [Dataset] (#dataset-CSN)
	- [Fine-tuning] (#finetuning)
		- [Dataset] (#dataset-JustInTime)
- [Citing CodeSync](#citing-codesync)
- [Contact Us](#contact-us)
- [License](#license) -->

___________
# CodeSync Package
This is a tool that uses for automated detection to identify inconsistencies between code and docstrings, and generates comprehensive replacements for inconsistent docstrings.

<p align="center">
  <img src="./assets/overview.png" width="300px" alt="overview">
</p>

## Getting Started

CodeSync can be easily to install and use as a Python package:

```bash
pip install codeSyncNet
```

## Inference

```python
from codeSyncNet import CodeSyncNet

model = CodeSyncNet()

code = """
    double sum2num(int a, int b) {
        return a + b;
    }
"""

docstring = """
    /**
    * Sum of 2 number
    * @param a int number
    * @param b int number
    */

"""
model.inference(code, docstring)
>>> MATCH!

code = """
    def func(self):
        self.caller.db.full_name = ""
        self.caller.msg("Full Name Cleared.")   
        self.menutree.goto("fullname")
"""

docstring = """
    Execute the command
"""

model.inference(code, docstring)
>>> UNMATCH!
>>> Recommended docstring: Clear the full_name.
```


## Pre-training 
### Installation
Setup environment and install dependencies
```bash
pip -r install requirements.txt
```

### Dataset
The dataset we used comes from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text).
It can be downloaded by following the command line:

```bash
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Text/code-to-text/dataset.zip
unzip dataset.zip
rm dataset.zip
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```

To reproduce the pretrained model, follow below command line:
```shell
python -m torch.distributed.run --nproc_per_node=2 run.py \
	--do_train \
	--do_eval \
	--task pretrain \
	--data_folder dataset/pretrain_dataset \ 
	--num_train_epochs 10 
```

## Fine-tuning 
To demonstrate the performance of our approach, we fine-tune CodeSync on the Just-In-Time task. The purpose of this task is to determine whether the comment is semantically out of sync with the corresponding code function.

### Dataset

Download data for the [Just-In-Time](https://github.com/panthap2/deep-jit-inconsistency-detection) task from [here](https://drive.google.com/drive/folders/1heqEQGZHgO6gZzCjuQD1EyYertN4SAYZ?usp=sharing)

We also provide fine-tune settings for CodeSync, whose results are reported in the paper.

```shell

# Training
python -m torch.distributed.run --nproc_per_node=2 run.py \
	--do_train \
	--do_eval \
	--post_hoc \
	--task just_in_time \
	--load_model \
	--data_folder dataset/just_in_time \ 
	--num_train_epochs 30 

# Testing
python -m torch.distributed.run --nproc_per_node=2 run.py \
	--do_test \
	--post_hoc \
	--task just_in_time \
	--data_folder dataset/just_in_time \ 
```

# Reference
More details can be found in our [paper](https://arxiv.org/abs/). 
If you use this code or our package, please consider citing us:

```bibtex
@article{codesync,
  title={Bootstrapping Code-Text Pretrained Language Model to Detect Inconsistency Between Code and Comment},
  author={},
  journal={},
  pages={},
  year={2023}
}
```

# Contact us
If you have any questions, comments or suggestions, please do not hesitate to contact us.
- Website: [fpt-aicenter](https://www.fpt-aicenter.com/ai-residency/)
- Email: support.ailab@fpt.com

# License
[MIT License](LICENSE.txt)