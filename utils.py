import os
import ast
from collections import deque
import random
import numpy as np
import torch
import multiprocessing
import json
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from model import Seq2Seq
from diff_utils import *
from statistics import mean, median
import matplotlib.pyplot as plt
from sklearn import metrics
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    cpu_cont = multiprocessing.cpu_count()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    device = torch.device("cuda", args.gpu)
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.cpu_cont = cpu_cont
    setup_for_distributed(args.rank == 0)     

def compute_score(predicted_labels, gold_labels):
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    assert(len(predicted_labels) == len(gold_labels))

    for i in range(len(gold_labels)):
        if gold_labels[i]:
            if predicted_labels[i]:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted_labels[i]:
                false_positives += 1
            else:
                true_negatives += 1
    
    if verbose:
        print('True positives: {}'.format(true_positives))
        print('False positives: {}'.format(false_positives))
        print('True negatives: {}'.format(true_negatives))
        print('False negatives: {}'.format(false_negatives))
    
    try:
        precision = true_positives/(true_positives + false_positives)
    except:
        precision = 0.0
    try:
        recall = true_positives/(true_positives + false_negatives)
    except:
        recall = 0.0
    try:
        f1 = 2*((precision * recall)/(precision + recall))
    except:
        f1 = 0.0

    return precision, recall, f1

def confusion_matrix(predict, label, path):
    
    confusion_matrix = metrics.confusion_matrix(label, predict)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

    cm_display.plot()
    plt.savefig(path+'/matrix.png')
    plt.show()

def build_or_load_gen(args):
    # build model
    if 'unixcoder' in args.model_name_or_path:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        # import！！！you must set is_decoder as True for generation

        config.is_decoder = True
        encoder = RobertaModel.from_pretrained(args.model_name_or_path,config=config) 

        tokenizer.add_tokens(["<REPLACE_OLD>", '<REPLACE_NEW>', "<REPLACE_END>", "<KEEP>","<KEEP_END>", "<INSERT_END>", "<DELETE_END>",
                                "<INSERT>", "<DELETE>","<INSERT_OLD_KEEP_BEFORE>", "<INSERT_NEW_KEEP_BEFORE>"],special_tokens=True)
        config.vocab_size = len(tokenizer)
        encoder.resize_token_embeddings(len(tokenizer))


        model = Seq2Seq(encoder=encoder,decoder=encoder,config=config,
                    beam_size=args.beam_size,max_length=args.max_target_length,
                    sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id,
                    queue_size=args.queue_size, device= args.device, ensemble=args.ensemble)

    return config, tokenizer, model

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_tqdm(iterator, desc=""):
        return iterator


def compute_score(predicted_labels, gold_labels, verbose=False):
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    assert(len(predicted_labels) == len(gold_labels))

    for i in range(len(gold_labels)):
        if gold_labels[i]==0:
            if predicted_labels[i]==0:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted_labels[i]==0:
                false_positives += 1
            else:
                true_negatives += 1
    
    if verbose:
        print('True positives: {}'.format(true_positives))
        print('False positives: {}'.format(false_positives))
        print('True negatives: {}'.format(true_negatives))
        print('False negatives: {}'.format(false_negatives))
    
    try:
        precision = true_positives/(true_positives + false_positives)
    except:
        precision = 0.0
    try:
        recall = true_positives/(true_positives + false_negatives)
    except:
        recall = 0.0
    try:
        f1 = 2*((precision * recall)/(precision + recall))
    except:
        f1 = 0.0
    return precision, recall, f1


def count_probability_justInTime():
    examples = []
    language = ['Summary','Return', 'Param']
    root_folder = './dataset/just_in_time/'
    stages = ['train', 'test', 'valid', 'test_clean']
    count_code = []
    count_nl = []
    for stage in stages:
        for lan in language:
            filename = root_folder + lan + '/' + stage + '.json'
            with open(filename, encoding="utf-8") as f:
                data = ast.literal_eval(f.read())

            for idx, js in enumerate(data):
                    code = ' '.join(js['span_diff_code_subtokens']).replace('\n', ' ')
                    code = ' '.join(code.strip().split())
                    nl = ' '.join(js['new_comment_subtokens']).replace('\n', '')
                    nl = ' '.join(nl.strip().split())
                    count_code.append(len(js['span_diff_code_subtokens']))
                    count_nl.append(len(js['new_comment_subtokens']))
    print('code max: ', max(count_code))
    print('code min: ', min(count_code))
    print('code mean: ', mean(count_code))
    print('code median: ', median(count_code))
    print('nl max: ', max(count_nl))
    print('nl min: ', min(count_nl))
    print('nl mean: ', mean(count_nl))
    print('nl median: ', median(count_nl))

def generate_diff_comment_justInTime():
    examples = []
    language = ['Summary','Return', 'Param']
    root_folder = './dataset/just_in_time/'
    stages = [ 'test','train','valid']
    count_nl = []
    count_pl = []
    for stage in stages:
        for lan in language:
            filename = root_folder  + lan + '/' + stage + '_old.json'
            with open(filename, encoding="utf-8") as f:
                data = ast.literal_eval(f.read())
            D = []
            for idx, js in enumerate(data):
                try:
                    nl_new = js["new_comment_subtokens"]
                    nl_old = js["old_comment_subtokens"]
                    diff_comment,_,_ = compute_comment_diffs(nl_old, nl_new)
                    count_nl.append(len(diff_comment))
                    count_pl.append(len(js['span_diff_code_subtokens']))
                    js["span_diff_comment_subtokens"] = diff_comment
                    print(' '.join(diff_comment))
                    D.append(js)
                except:
                    None
            with open(root_folder + lan + '/' + stage + '.json', 'w') as fo:
                json.dump(D, fo)
