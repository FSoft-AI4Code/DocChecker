<div align="center">

<p align="center">
  <img src="./assets/logo.jpg" width="300px" alt="logo">
</p>
	
  <a href="https://opensource.org/license/apache-2-0/">
  <img alt="license" src="https://img.shields.io/badge/License-Apache%202.0-green.svg"/>
  </a>
   <a href="https://www.python.org/downloads/release/python-380/">
  <img alt="python" src="https://img.shields.io/badge/python-3.8+-yellow.svg"/>
  </a> 
   <a href="https://pypi.org/project/docchecker/">
  <img alt="downloads" src="https://static.pepy.tech/badge/salesforce-codetf"/>
  </a> 
	
# DocChecker: Bootstrapping Code-Text Pretrained Language Model to Detect Inconsistency Between Code and Comment

</div>

# Table of content
- [Introduction](#introduction)
- [Installation](#installation-guide)
- [Getting Started](#getting-started)
	- [Inferencing Pipeline](#inferencing-pipeline)
	- [Pre-training Pipeline](#pre-training-pipeline)
		- [Installation for Pre-training](#installation-for-pre-training)
	 	- [Dataset for Pre-training](#dataset-for-pre-training)
	- [Fine-tuning Pipeline](#fine-tuning-pipeline)
		- [Dataset for Fine-tuning](#dataset-for-fine-tuning)
- [Playground](#playground)
- [Citing Us](#citing-us)
- [Contact Us](#contact-us)
- [License](#license) 

___________
# Introduction
Comments on source code serve as critical documentation for enabling developers to understand the code's functionality and use it properly. However, it is challenging to ensure that comments accurately reflect the corresponding code, particularly as the software evolves over time. Although increasing interest has been taken in developing automated methods for identifying and fixing inconsistencies between code and comments, the existing methods have primarily relied on heuristic rules. 

DocChecker is trained on top of encoder-decoder model to learn from code-text pairs. It is jointly pre-trained with three objectives: code-text contrastive learning, binary classification, and text generation. DocChecker is a tool that be used to detect noisy code-comment pairs and generate synthetic comments, enabling it to determine comments that do not match their associated code snippets and correct them.
Its effectiveness is demonstrated on the Just-In-Time dataset compared with other state-of-the-art methods. 

<p align="center">
  <img src="./assets/overview.png" width="800px" alt="overview">
</p>

# Installation Guide

1. (Optional) Creating conda environment

```bash
conda create -n docchecker python=3.8
conda activate docchecker
```

2. Install from [PyPI](https://pypi.org/project/docchecker/):
```bash
pip install docchecker
```
    
3. Alternatively, build DocChecker from source:

```bash
git clone https://github.com/FSoft-AI4Code/DocChecker.git
cd DocChecker
pip install -r requirements.txt .
```

# Getting Started
## Inferencing pipeline

Getting started with DocChecker is simple and quick with our tool by using ``inference()`` function. 

```python
from DocChecker.utils import inference
```
There are a few notable arguments that need to be considered:

Parameters:

- input_file_path (str): the file path that contains source code, if you want to check all the functions in there.
- raw_code (str): a sequence of source code if `input_file_path` is not given.
- language (str, required): the programming language that corresponds your raw_code. We support 10 popular programming languages, including Java, JavaScript, Python, Ruby, Rust, Golang, C#, C++, C, and PHP.
- output_file_path (str): if `output_file_path` is given, the results from our tool will be written in `output_file_path`; otherwise, they will be printed on the screen.

Returns:

- list of dictionaries, including:
    - function_name: the name of each function in the raw code
    - code: code snippet
    - docstring: the docstring corresponding code snippet
    - predict: the prediction of DocChecker. It returns “Inconsistent!” for the inconsistent pair and “Consistent!” means the docstring is consistent with the code in a code-text pair
    - recommend_docstring: If a code-text pair is considered as “Inconsistent!”, DocChecker will replace its docstring by giving comprehensive ones; otherwise, it will keep the original version.

Here's an example showing how to load docchecker model and perform inference on inconsistent detection task:

```python
from DocChecker.utils import inference

code = """
            def inject_func_as_unbound_method(class_, func, method_name=None):
                # This is actually quite simple
                if method_name is None:
                    method_name = get_funcname(func)
                setattr(class_, method_name, func)

            def e(message, exit_code=None):
                # Print an error log message.
                print_log(message, YELLOW, BOLD)
                if exit_code is not None:
                    sys.exit(exit_code)
        """

inference(raw_code=code,language='python')

>>[
    {
    "function_name": "inject_func_as_unbound_method",
    "code": "def inject_func_as_unbound_method(class_, func, method_name=None):\n    \n    if method_name is None:\n        method_name = get_funcname(func)\n    setattr(class_, method_name, func)",
    "docstring": " This is actually quite simple",
    "predict": "Inconsistent!",
    "recommended_docstring": "Inject a function as an unbound method."
    },
    {
        "function_name": "e",
        "code": "def e(message, exit_code=None):\n    \n    print_log(message, YELLOW, BOLD)\n    if exit_code is not None:\n        sys.exit(exit_code)",
        "docstring": "Print an error log message.",
        "predict": "Consistent!",
        "recommended_docstring": "Print an error log message."
    }
]
```

## Pre-training Pipeline
We also provide our source code for you to re-pretraining DocChecker.

### Installation for Pre-training
Setup environment and install dependencies for pre-training.
```bash
cd ./DocChecker
pip -r install requirements.txt
```

### Dataset for Pre-training
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

To re-pretrain, follow the below command line:
```shell
python -m torch.distributed.run --nproc_per_node=2 run.py \
	--do_train \
	--do_eval \
	--task pretrain \
	--data_folder dataset/pretrain_dataset \ 
	--num_train_epochs 10 
```

## Fine-tuning Pipeline
To demonstrate the performance of our approach, we fine-tune DocChecker on the Just-In-Time task. The purpose of this task is to determine whether the comment is semantically out of sync with the corresponding code function.

### Dataset for Fine-tuning

Download data for the [Just-In-Time](https://github.com/panthap2/deep-jit-inconsistency-detection) task from [here].(https://drive.google.com/drive/folders/1heqEQGZHgO6gZzCjuQD1EyYertN4SAYZ?usp=sharing)

We also provide fine-tune settings for DocChecker, whose results are reported in the paper.

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

# Playground
We provide an interface for DocChecker at the [link](http://4.193.50.237:5000/).
The demostration can be found at [Youtube](https://youtu.be/KFbyaSf2I3c).

# Citing Us
More details can be found in our [paper](https://arxiv.org/abs/). 
If you use this code or our package, please consider citing us:

```bibtex
@article{DocChecker,
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
[Apache License Version 2.0](LICENSE.txt)
