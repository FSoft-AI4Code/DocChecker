import json
import ast
import random
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from os import listdir
from os.path import isfile, join
from utils import get_tqdm
import os


class Example(object):
    """A single training/test example."""
    """
     Constructor for the index source target and label
     
     @param self - the index
     @param idx - Index of the target.
     @param source - Source of the target.
     @param target - Target of the node.
     @param label - Label of the target
    """

    def __init__(self,
                 idx,
                 source,
                 target,
                 label,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.label = label

# ----------------------------------
"""
 Read examples from filename.
 
 @param filename - file to read examples from.
 @param stage - the stage of the example to read.
"""


def read_examples(filename, args, stage='train'):
    if args.task == 'just_in_time':
        return read_examples_justInTime(filename, stage, args, post_hoc=args.post_hoc)
    elif args.task == 'pretrain':
        return read_examples_CSN(filename, stage, args)
    elif args.task == 'cup':
        return read_examples_cup(filename, stage, args)


def read_examples_cup(root_folder, stage, args):
    """Read examples from filename."""
    examples = []
    filename = root_folder +  stage + '.jsonl'
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = js['diff_code_change'].replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = js['dst_desc'].replace('\n', '')
            nl = ' '.join(nl.strip().split())
            # if 'label' in js:
            #     label = js['label']
            # else:
            label = 0
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                    label=label
                )
            )
    return examples


def read_examples_CSN(root_folder, stage, args):
    """Read examples from filename."""
    examples = []
    for lan in args.language:
        filename = root_folder + lan + '/' + stage + '.jsonl'
        with open(filename, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                if 'idx' not in js:
                    js['idx'] = idx
                code = ' '.join(js['code_tokens']).replace('\n', ' ')
                code = ' '.join(code.strip().split())
                nl = ' '.join(js['docstring_tokens']).replace('\n', '')
                nl = ' '.join(nl.strip().split())
                if 'label' in js:
                    label = js['label']
                else:
                    label = 0
                examples.append(
                    Example(
                        idx=idx,
                        source=code,
                        target=nl,
                        label=label
                    )
                )
    return examples

def read_examples_justInTime(root_folder, stage, args, post_hoc=None):
    """Read examples from filename."""
    examples = []
    for lan in args.language:
        filename = root_folder + lan + '/' + stage + '.json'
        with open(filename, encoding="utf-8") as f:
            data = ast.literal_eval(f.read())

        for idx, js in enumerate(data):
            if post_hoc:
                code = ' '.join(js['new_code_subtokens']).replace('\n', ' ')
                code = ' '.join(code.strip().split())
                nl = ' '.join(js['old_comment_subtokens']).replace('\n', '')
                nl = ' '.join(nl.strip().split())
            else:
                code = ' '.join(js['span_diff_code_subtokens']).replace('\n', ' ')
                code = ' '.join(code.strip().split())
                nl = ' '.join(js['old_comment_subtokens']).replace('\n', '')
                nl = ' '.join(nl.strip().split())
            label = (js['label']+1)%2

            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                    label=label,
                )
            )
    return examples

def read_examples_infer(root_folder, stage, args):
    """Read examples from filename."""
    examples = []
    raw_examples = []
    for lan in args.language:
        filename = root_folder + lan + '/' + stage + '.jsonl'
        with open(filename, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                # if idx == 100:
                #     break
                line = line.strip()
                js = json.loads(line)
                raw_examples.append(js)
                if 'idx' not in js:
                    js['idx'] = idx
                code = ' '.join(js['code_tokens']).replace('\n', ' ')
                code = ' '.join(code.strip().split())
                nl = ' '.join(js['docstring_tokens']).replace('\n', '')
                nl = ' '.join(nl.strip().split())
                if 'label' in js:
                    label = js['label']
                else:
                    label = 0
                examples.append(
                    Example(
                        idx=idx,
                        source=code,
                        target=nl,
                        label=label
                    )
                )
    return examples, raw_examples


class InputFeatures(object):
    """A single training/test features for a example."""
    """
     A class that represents a single example.
     
     @param self - the example
     @param example_id - example id
     @param source_ids - Source IDs.
     @param target_ids - Target ids.
     @param label - Label of the target.
    """

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 label,
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.label = label

"""
 convert examples to token ids
 
 @param examples - list of examples to convert to token ids
 @param tokenizer - the tokenizer to use.
 @param args - the arguments to pass to the feature decoder.
 @param stage - the stage of the feature to convert to token ids
"""


def convert_examples_to_features(item):
    """convert examples to token ids"""
    example_index, example,  tokenizer, args = item
  
    if example_index % 5000 == 0:
        print(example_index)
    if 'unixcoder' in args.model_name_or_path:
        source_tokens = tokenizer.tokenize(example.source)[
            :args.max_source_length-5]
        source_tokens = [tokenizer.cls_token, "<encoder-decoder>",
                            tokenizer.sep_token, "<mask0>"]+source_tokens+[tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id]*padding_length

        target_tokens = tokenizer.tokenize(example.target)[
            :args.max_target_length-2]
        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length

   
    else:
        source_str = example.source.replace('</s>', '<unk>')
        source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
        assert source_ids.count(tokenizer.eos_token_id) == 1
       
        target_str = example.target
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                    truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1


    label = example.label


    return InputFeatures(
            example_index,
            source_ids,
            target_ids,
            label,
        )


def get_dataloader(args, filename, tokenizer, pool, stage='train', label=False, sequential=False, num_sample=None, infer=False,lan=None):

    if infer:
        examples, raw_examples = read_examples_infer(filename, stage=stage, args=args)
    else:
        examples = read_examples(filename, args, stage=stage)

    if num_sample != None:
        examples = random.sample(examples, min(num_sample, len(examples)))

    print('Reading raw samples has done !!!')
    print(len(examples))
    tuple_examples = [(idx, example,  tokenizer, args) for idx, example in enumerate(examples)]

    features = pool.map(convert_examples_to_features, get_tqdm(tuple_examples))
    all_source_ids = torch.tensor(
        [f.source_ids for f in features], dtype=torch.long)
    all_target_ids = torch.tensor(
        [f.target_ids for f in features], dtype=torch.long)

    print('Converting raw samples to ids has done !!!')

    if label:
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_target_ids, all_label)
        
        data = TensorDataset(all_source_ids, all_target_ids, all_label)
    else:
        data = TensorDataset(all_source_ids, all_target_ids)

    if sequential:
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)

    if 'test' in stage:
        drop_last = False
    else:
        drop_last = True
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size //
                            args.gradient_accumulation_steps, drop_last=drop_last)

    print('Creating dataloader has done !!!')

    if infer:
        return examples, dataloader, raw_examples
    else:
        return examples, dataloader
