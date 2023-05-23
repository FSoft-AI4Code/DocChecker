<div align="center">

<p align="center">
  <img src="./assets/logo.jpg" width="300px" alt="logo">
</p>

# DocChecker: Bootstrapping Code-Text Pretrained Language Model to Detect Inconsistency Between Code and Comment

</div>

<!-- ## Table of content
- [DocChecker package](#DocChecker-package)
	- [Getting Started] (#getting-started)
	- [Inference] (#inference)
	- [Pre-training] (#pretrain)
		- [Installation] (#install)
	 	- [Dataset] (#dataset-CSN)
	- [Fine-tuning] (#finetuning)
		- [Dataset] (#dataset-JustInTime)
- [Citing DocChecker](#citing-DocChecker)
- [Contact Us](#contact-us)
- [License](#license) -->

___________
# The DocChecker Tool
DocChecker is trained on top of encoder-decoder model to learn from code-text pairs. It is a tool that uses for automated detection to identify inconsistencies between code and docstrings, and generates comprehensive summary sentence to replace the old ones.

<p align="center">
  <img src="./assets/overview.png" width="800px" alt="overview">
</p>

## Usage Scenario

### Installation
Install the dependencies:

```bash
pip -r install requirements.txt
```

### Inference

Since DocChecker is a Python package, users can use it by `inference` function. 

```python
from DocChecker.utils import inference
```
Parameters:

- input_file_path (str): the file path that contains source code, if you want to check all the functions in there.
- raw_code (str): a sequence of source code if `input_file_path` is not given.
- language (str, required): the programming language. We support 10 popular programming languages such as Java, JavaScript, Python, Ruby, Rust, Golang, C#, C++, C, and PHP.
- output_file_path (str): if `input_file_path` is given, the results from our tool will be written in `output_file_path`; otherwise, they will be printed on the screen.

Returns:

- list of dictionaries, including:
    - function_name: the name of each function in the raw code
    - code: code snippet
    - docstring: the docstring corresponding code snippet
    - predict: the prediction of DocChecker. It returns “Inconsistent!” for the inconsistent pair and “Consistent!” means the docstring is consistent with the code in a code-text pair
    - recommend_docstring: If a code-text pair is considered as “Inconsistent!”, DocChecker will replace its docstring by giving comprehensive ones; otherwise, it will keep the original version.

#### Example
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

## Pre-training 
### Installation
Setup environment and install dependencies
```bash
cd ./DocChecker
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

To reproduce the pre-trained model, follow below command line:
```shell
python -m torch.distributed.run --nproc_per_node=2 run.py \
	--do_train \
	--do_eval \
	--task pretrain \
	--data_folder dataset/pretrain_dataset \ 
	--num_train_epochs 10 
```

## Fine-tuning 
To demonstrate the performance of our approach, we fine-tune DocChecker on the Just-In-Time task. The purpose of this task is to determine whether the comment is semantically out of sync with the corresponding code function.

### Dataset

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

# Reference
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
[MIT License](LICENSE.txt)