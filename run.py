# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import bleu
import torch
import logging
import argparse
import numpy as np
from io import open
from tqdm import tqdm
import utils
import wandb
from dataloader import *
import multiprocessing

from transformers import (AdamW, get_linear_schedule_with_warmup)
from statistics import mean

logger = logging.getLogger(__name__)

        
def main():
    parser = argparse.ArgumentParser()
 
    ## Required parameters  
    parser.add_argument("--model_name_or_path", default='microsoft/unixcoder-base', type=str,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default='./saved_model/', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")   
    parser.add_argument("--load_model_dir", default='./saved_model/pretrained_model', type=str, 
                        help="The output directory where the pretrained model was saved")   
  
    ## Other parameters
    parser.add_argument("--output_clean_dir", type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")   

    parser.add_argument("--task", default='just_in_time', type=str,
                        choices=['pretrain', 'just_in_time'],)   

    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="The folder that contains dataset")
   
    parser.add_argument("--run_name", default='', type=str, 
                        help="name for each running in wandb")  
    parser.add_argument("--max_source_length", default=200, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--wandb", action='store_true',
                        help="whether to visualize training phase by wandb")
    parser.add_argument("--post_hoc", action='store_true',
                        help="whether to run the setting of post hoc")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    parser.add_argument("--load_model", action='store_true',
                        help="Whether to load the pretrained checkpoint.")
    
    parser.add_argument("--train_batch_size", default=100, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--alpha", default=1/3, type=float,
                        help="hyperparam in loss function for language model loss.")
    parser.add_argument("--beta", default=1/3, type=float,
                        help="hyperpapram in loss function for contrastive learning loss.")
    parser.add_argument("--queue_size", default=57600, type=int,
                        help="size for the queue in the model.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=0.00005, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=4, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.") 
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--distributed',  action='store_true')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # print arguments    
    args = parser.parse_args()
    
    #set dataset
    if args.task == 'just_in_time':
        args.language = ['Summary','Return', 'Param']
        if args.post_hoc:
            args.output_dir += 'post_hoc/'
        else: 
            args.output_dir += 'just_in_time/'
        args.num_train_epochs = 30
    elif args.task == 'pretrain':
        args.language = [ 'go', 'java', 'javascript', 'php', 'python', 'ruby']
        args.output_dir += 'pretrained_CSN/'

    if args.wandb:
        wandb.init(project="CodeSyncNet", name = args.run_name)

    # set log

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    logging.basicConfig(filename=args.output_dir + '/run.log',
                     filemode='a',
                    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
    # set device    
    utils.init_distributed_mode(args) 
    
    if args.wandb:
        wandb.config = {
            "learning_rate": args.learning_rate,
            "epochs": args.num_train_epochs,
            "batch_size": args.train_batch_size,
            "beam_size": args.beam_size
        }
    
    # Set seed
    utils.set_seed(args.seed)

    pool = multiprocessing.Pool(args.cpu_cont)
    scaler = torch.cuda.amp.GradScaler()

    config, tokenizer, model = utils.build_or_load_gen(args)

    logger.info("Training/evaluation parameters %s", args)
    
    map_location = {"cuda:0": "cuda:%d" % args.rank} if args.distributed else None
    if args.distributed:
        model.cuda()
        if args.load_model:
            checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
            output_dir = os.path.join(args.load_model_dir, checkpoint_prefix)  
            model.load_state_dict(torch.load(output_dir,map_location='cuda:0'))
    else:
        model.to(args.device)
        if args.load_model:
            checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
            output_dir = os.path.join(args.load_model_dir, checkpoint_prefix)  
            model.load_state_dict(torch.load(output_dir, map_location='cuda:0'))
    model.queue_ptr[0] = 0
    if args.do_train:
        # Prepare training data loader
        if args.task == 'just_in_time':
            train_examples, train_dataloader = get_dataloader(args, args.data_folder, tokenizer=tokenizer, pool=pool, stage='train', label=True)
        elif args.task == 'pretrain':
            train_examples, train_dataloader = get_dataloader(args, args.data_folder, tokenizer=tokenizer, pool=pool,stage='train')
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=int(len(train_dataloader)*args.num_train_epochs*0.1),
                                                    num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples)) 
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        

        model.train()
        patience, best_bleu, best_loss, best_acc, best_f1, losses, dev_dataset = 0, 0, 10000, 0, 0, [], {}
        losses_lm = []
        losses_ita = []
        losses_itm = []

        for epoch in range(0, args.num_train_epochs):
            total_num = 0
            total_acc = 0
            for batch in tqdm(train_dataloader):
                
                if args.task == 'just_in_time':
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, target_ids, label= batch
                    
                    total_num += source_ids.size(0)
                    loss_lm, loss_ita, loss_itm, hits = model(source_ids=source_ids,target_ids=target_ids, labels=label, just_in_time=True)
                    
                    total_acc += hits.sum().data.cpu().numpy()
                elif args.task == 'pretrain':
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids,target_ids = batch[0], batch[1]
                   
                    total_num += 2*source_ids.size(0)
                    loss_lm, loss_ita, loss_itm, hits = model(source_ids=source_ids,target_ids=target_ids)
                   
                    total_acc += hits.sum().data.cpu().numpy()
                

                loss = args.alpha*loss_lm + args.beta*loss_ita + (1-args.alpha-args.beta)*loss_itm

                if args.wandb:
                    wandb.log({
                        "tota loss": loss, 
                        "loss lm": loss_lm,
                        "loss contrastive": loss_ita,
                        "loss binary classification": loss_itm,
                        "train acc": total_acc/total_num*100})

                # breaknum_train_epochs
                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                losses.append(loss.item())
                losses_ita.append(loss_ita.item())
                losses_itm.append(loss_itm.item())
                losses_lm.append(loss_lm.item())
                loss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    if len(losses) // args.gradient_accumulation_steps % 100 == 0:
                        logger.info("epoch {} step {} total loss {} loss_lm {} loss_contrastive {} loss_binary {} acc {:.2f}".format(epoch,
                                                     len(losses)//args.gradient_accumulation_steps,
                                                     round(np.mean(losses[-100*args.gradient_accumulation_steps:]),4),
                                                     round(np.mean(losses_lm[-100*args.gradient_accumulation_steps:]),4), 
                                                     round(np.mean(losses_ita[-100*args.gradient_accumulation_steps:]),4),
                                                     round(np.mean(losses_itm[-100*args.gradient_accumulation_steps:]),4),
                                                     total_acc/total_num*100))
            
                    acc = total_acc/total_num*100
            
                    if len(losses) // args.gradient_accumulation_steps % 5000 == 0 and args.do_eval and args.task == 'pretrain':
                        #Eval model with dev dataset                   
                        if 'dev' in dev_dataset:
                            eval_examples, eval_dataloader = dev_dataset['dev']
                        else:
                            eval_examples, eval_dataloader = get_dataloader(args, args.data_folder, pool=pool,tokenizer=tokenizer, stage='valid', label=False, sequential=True, num_sample=1000)
                            dev_dataset['dev']= eval_examples, eval_dataloader

                        logger.info("\n***** Running evaluation *****")
                        logger.info("  Num examples = %d", len(eval_examples))
                        logger.info("  Batch size = %d", args.eval_batch_size)
                        losses_eval = []

                        model.eval() 
                        p=[]
                        # pred_ids = []
                        for batch in eval_dataloader:
                            batch = tuple(t.to(args.device) for t in batch)
                            source_ids = batch[0]                  
                            target_ids = batch[1]
                            with torch.no_grad():
                                loss_lm, loss_ita,loss_itm, pred_sentence = model(source_ids=source_ids,target_ids=target_ids, stage='dev') 
                                for pred in pred_sentence:
                                    t = pred[0].cpu().numpy()
                                    t = list(t)
                                    if 0 in t:
                                        t = t[:t.index(0)]
                                    text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                                    p.append(text)

                                loss = args.alpha*loss_lm + args.beta*loss_ita + (1-args.alpha-args.beta)*loss_itm         
                                losses_eval.append(loss.item())
                                
                        model.train()
                        predictions = []
                        with open(args.output_dir+"/dev.output",'w') as f, open(args.output_dir+"/dev.gold",'w') as f1:
                            for ref,gold in zip(p,eval_examples):
                                predictions.append(str(gold.idx)+'\t'+ref)
                                f.write(str(gold.idx)+'\t'+ref+'\n')
                                f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

                        (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold")) 
                        dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                        logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                        logger.info("  "+"*"*20)    
                        if dev_bleu > best_bleu:
                            logger.info("  Best bleu:%s",dev_bleu)
                            logger.info("  "+"*"*20)
                            best_bleu = dev_bleu
                            # Save best checkpoint for best bleu
                            output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            patience =0
                        else:
                            patience +=1
                            if patience == 5:
                                break

                        if np.mean(losses_eval) < best_loss:
                            logger.info("  Best loss:%s", np.mean(losses_eval))
                            logger.info("  "+"*"*20)
                            best_loss = np.mean(losses_eval)
                            # Save best checkpoint for best bleu
                            output_dir = os.path.join(args.output_dir, 'checkpoint-best-loss')
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            patience =0
                            
                        
                        
    
            if args.task == 'pretrain':
                output_dir = os.path.join(args.output_dir, 'checkpoint-epoch-{}'.format(epoch))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
            elif args.task == 'just_in_time':
                if 'dev' in dev_dataset:
                    eval_examples, eval_dataloader = dev_dataset['dev']
                else:
                    eval_examples, eval_dataloader = get_dataloader(args, args.data_folder, pool=pool,tokenizer=tokenizer, stage='valid', label=True)
                    dev_dataset['dev']= eval_examples, eval_dataloader

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                model.eval() 
                p=[]

                total_num = 0
                total_acc = 0
                target_labels = []
                pred_labels = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, target_ids, labels = batch  
                    
                    bs = source_ids.size(0)         
                    with torch.no_grad():
                        pred, hits = model(source_ids, target_ids, labels=labels, stage='test')  
                        total_num += bs
                        total_acc += hits.sum().data.cpu().numpy()
                        target_labels.extend(labels.tolist())
                        pred_labels.extend(pred.tolist())
                acc = total_acc/total_num*100
                precision, recall, F1_score = utils.compute_score(pred_labels, target_labels)
                model.train()
                if acc > best_acc:
                    logger.info("  Best acc:%s", acc)
                    logger.info("  "+"*"*20)
                    best_acc = acc
                    # Save best checkpoint for best acc
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    patience = 0 
                else:
                    patience += 1
                    if patience == 8:
                        break
                
                if F1_score > best_f1:
                    logger.info("  Best F1:%s", F1_score)
                    logger.info("  "+"*"*20)
                    best_f1 = F1_score
                    # Save best checkpoint for best F1 score
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-F1')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    patience = 0 
            

            if args.task == 'just_in_time' and args.do_test:
                if 'test' in dev_dataset:
                    eval_examples, eval_dataloader = dev_dataset['test']
                else:
                    eval_examples, eval_dataloader = get_dataloader(args, args.data_folder, tokenizer=tokenizer, pool=pool,stage='test', label=True, sequential=True)
                    dev_dataset['test'] = eval_examples, eval_dataloader

                logger.info("\n***** Running evaluation on the synthetic  *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                model.eval() 
                p=[]
                total_num = 0
                total_acc = 0
                target_labels = []
                pred_labels = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, target_ids, labels = batch  
                    bs = source_ids.size(0)         
                    with torch.no_grad():
                        pred_label, hits = model(source_ids, target_ids, labels=labels, stage='test')  
                       
                        total_num += bs
                        total_acc += hits.sum().data.cpu().numpy()
                        target_labels.extend(labels.tolist())
                        pred_labels.extend(pred_label.tolist())
                acc = total_acc/total_num*100
                precision, recall, f1 = utils.compute_score(pred_labels, target_labels)
                logger.info(' Testing ACC = {:.2f}'.format(acc))
                logger.info(' Recall score = {:.3f}'.format(recall))
                logger.info(' Precision score = {:.3f}'.format(precision))
                logger.info(' F1 score= {:.3f}'.format(f1))

    
    if args.do_test:
        if args.task == 'just_in_time':
            checkpoint_prefix = 'checkpoint-best-acc/pytorch_model.bin'
            eval_examples, eval_dataloader = get_dataloader(args, args.data_folder, tokenizer=tokenizer, pool=pool,stage='test', label=True, sequential=True)
        else:
            checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
            eval_examples, eval_dataloader = get_dataloader(args, args.data_folder, pool=pool,tokenizer=tokenizer, stage='train', label=False, sequential=True, num_sample=1000)

        output_dir = os.path.join(args.output_dir, checkpoint_prefix)  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir,map_location='cuda:0')  )  
        model.eval()

        logger.info("\n***** Running evaluation on the test set *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        if args.task == 'just_in_time':
            model.eval() 
            p=[]
            total_num = 0
            total_acc = 0
            target_labels = []
            pred_labels = []
            for batch in eval_dataloader:
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids, labels = batch  
               
                bs = source_ids.size(0)         
                with torch.no_grad():
                    pred, hits = model(source_ids, target_ids, labels=labels, stage='test')  
                    
                    total_num += bs
                    total_acc += hits.sum().data.cpu().numpy()
                    target_labels.extend(labels.tolist())
                    pred_labels.extend(pred.tolist())
            acc = total_acc/total_num*100
            precision, recall, f1 = utils.compute_score(pred_labels, target_labels)
            utils.confusion_matrix(pred_labels, target_labels, args.output_dir)
            logger.info(' Testing ACC = {:.2f}'.format(acc))
            logger.info(' Recall score = {:.3f}'.format(recall))
            logger.info(' Precision score = {:.3f}'.format(precision))
            logger.info(' F1 score= {:.3f}'.format(f1))
            with open(args.output_dir+"/test.output",'w') as f:                
                for label,target,gold in zip(pred_labels,target_labels,eval_examples):
                    f.write(str(gold.idx)+'\t'+str(label)+'\t'+str(target)+'\n')                        
        elif args.task == "pretrain":
            p = []    
            labels=[]
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids = batch[0]   
                target_ids = batch[1]               
               
               
                with torch.no_grad():
                    pred_labels, preds = model(source_ids,target_ids=target_ids, stage='inference')   
                    for label, pred in zip(pred_labels,preds):
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text)
                        labels.append(int(label))
                    
        
            model.train()
            predictions=[]
            with open(args.output_dir+"/test.output",'w') as f, open(args.output_dir+"/test.gold",'w') as f1:
                # test original
                
                for ref,label,gold in zip(p,labels,eval_examples):
                        predictions.append(str(gold.idx)+'\t'+str(label)+'\t'+ref)
                        f.write(str(gold.idx)+'\t'+str(label)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+str(label)+'\t'+gold.target+'\n')   

if __name__ == "__main__":
    main()
   

