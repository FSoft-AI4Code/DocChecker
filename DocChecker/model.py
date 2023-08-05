# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder, config, beam_size=4, max_length=32, sos_id=None, eos_id=None, queue_size=57600,
                    momentum = 0.995, embed_dim=256, device='cuda'):
        super(Seq2Seq, self).__init__()

        self.device=device

        self.encoder = encoder
        self.decoder = decoder

        self.encoder_m = encoder
        self.decoder_m = decoder

        self.encoder_proj = nn.Linear(config.hidden_size, embed_dim)
        self.decoder_proj = nn.Linear(config.hidden_size, embed_dim)
        self.encoder_proj_m = nn.Linear(config.hidden_size, embed_dim)
        self.decoder_proj_m = nn.Linear(config.hidden_size, embed_dim)

        self.config=config
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        
        self.itm_head = nn.Linear(config.hidden_size, 2) 
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)

        self.model_pairs = [[self.encoder,self.encoder_m],
                            [self.encoder_proj, self.encoder_proj_m],
                            [self.decoder,self.decoder_m],
                            [self.decoder_proj, self.decoder_proj_m]
                           ]   
        self.copy_params()
        self.register_buffer("code_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long)) 
        self.code_queue = nn.functional.normalize(self.code_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id       
        
    def forward(self, source_ids, target_ids=None, labels=None, stage=None, alpha=0.01, just_in_time=False, source_text_ids=None):   

        # print(self.queue_ptr)
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        if  stage =='test_original':
            return self.generate(source_ids)
        elif stage == 'dev' or stage=='get_positive':
            gen_sentence = self.generate(source_ids)


        mask_source = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        encoder_output = self.encoder(source_ids,attention_mask=mask_source,use_cache=True)  

        # get mask for output of encoder
        mask_encoder = source_ids.ne(1)
        mask_encoder = torch.unsqueeze(mask_encoder,-1)
        mask_encoder = mask_encoder.expand(-1, -1, self.config.hidden_size)
        encoder_output_contrastive = encoder_output.last_hidden_state*mask_encoder

        code_embeds = torch.mean(encoder_output_contrastive, dim=1)
        code_feat = F.normalize(self.encoder_proj(code_embeds), dim=-1)

        if source_text_ids != None:
            TARGET = target_ids
            target_ids = source_text_ids
        else:
            TARGET =  target_ids
            
        ids = torch.cat((source_ids,target_ids),-1)
        mask = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
        mask = mask & ids[:,None,:].ne(1)
        out = self.decoder(target_ids,attention_mask=mask,past_key_values=encoder_output.past_key_values).last_hidden_state

        # get mask for output of decoder
        mask_decoder = target_ids.ne(1)
        mask_decoder = torch.unsqueeze(mask_decoder,-1)
        mask_decoder = mask_decoder.expand(-1, -1, self.config.hidden_size)
        decoder_output_contrastive = out*mask_decoder

        # text_embeds = out[:, 0, :]
        text_embeds = torch.mean(decoder_output_contrastive, dim=1)
        text_feat = F.normalize(self.decoder_proj(text_embeds), dim=-1)

        sim_pos = code_feat @ text_feat.t() / self.temp
        if stage == 'get_positive':
            pred_output = self.itm_head(text_embeds)
            return gen_sentence, sim_pos, pred_output

        elif stage=='inference':
            pred_output = self.itm_head(text_embeds)
            _, pred = pred_output.max(1)
            pred = torch.tensor(pred, dtype=torch.int64)
            return  pred, self.generate(source_ids)

        elif stage == 'test':
            pred_output = self.itm_head(text_embeds)
            _, pred = pred_output.max(1)
            pred = torch.tensor(pred, dtype=torch.int64)
            hits = (pred == labels).float()
            return  pred, hits#, self.generate(source_ids)
       
        # ============= loss lm ====================

        ids_lm = torch.cat((source_ids,TARGET),-1)
        mask_lm = self.bias[:,source_ids.size(-1):ids_lm.size(-1),:ids_lm.size(-1)].bool()
        mask_lm = mask_lm & ids_lm[:,None,:].ne(1)
        out_lm = self.decoder(TARGET,attention_mask=mask_lm,past_key_values=encoder_output.past_key_values).last_hidden_state
        lm_logits = self.lm_head(out_lm)
        # Shift so that tokens < n predict n
        active_loss = TARGET[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = TARGET[..., 1:].contiguous()
     
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss_lm = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])
        
        
        # ============= loss contrastive =================== 
        with torch.no_grad():
            self._momentum_update()
            encoder_output_m = self.encoder_m(source_ids,attention_mask=mask_source,use_cache=True)
            code_embeds_m = torch.mean(encoder_output_m.last_hidden_state*mask_encoder, dim=1)
            code_feat_m = F.normalize(self.encoder_proj_m(code_embeds_m),dim=-1)  
            code_feat_all = torch.cat([code_feat_m.t(),self.code_queue.clone().detach()],dim=1)                   
            
            output_m = self.decoder_m(target_ids,attention_mask=mask,past_key_values=encoder_output_m.past_key_values).last_hidden_state  
            text_output_m = torch.mean(output_m*mask_decoder, dim=1)  
            text_feat_m = F.normalize(self.decoder_proj_m(text_output_m),dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = code_feat_m @ text_feat_all / self.temp  
            sim_t2i_m = text_feat_m @ code_feat_all / self.temp 

            sim_targets = torch.zeros(sim_i2t_m.size()).to(self.device)
            sim_targets.fill_diagonal_(1)          

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = code_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ code_feat_all / self.temp
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        self._dequeue_and_enqueue(code_feat_m, text_feat_m)        

        #============== code-text Matching ===================###
        
        # forward the positve code-text pair
        bs = source_ids.size(0)
        output_pos = text_embeds
        with torch.no_grad():       
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)+1e-4 
            weights_t2i.fill_diagonal_(0)            
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)+1e-4  
            weights_i2t.fill_diagonal_(0)   

        # select a negative text for each code

        
        text_ids_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(target_ids[neg_idx])
        

        text_ids_neg = torch.stack(text_ids_neg,dim=0)   
        text_ids_neg = text_ids_neg.to(torch.long)

        ids = torch.cat((source_ids,text_ids_neg),-1)
        mask = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
        mask = mask & ids[:,None,:].ne(1)

        output_neg = self.decoder(text_ids_neg,attention_mask=mask,past_key_values=encoder_output.past_key_values).last_hidden_state      

        mask_decoder_neg = text_ids_neg.ne(1)
        mask_decoder_neg = torch.unsqueeze(mask_decoder_neg,-1)
        mask_decoder_neg = mask_decoder_neg.expand(-1, -1, self.config.hidden_size)
        decoder_output_neg = output_neg*mask_decoder_neg
        text_embeds_neg = torch.mean(decoder_output_neg, dim=1)

        if just_in_time:
            vl_output = self.itm_head(output_pos)
            itm_labels = labels
        else:
            vl_embeddings = torch.cat([output_pos, text_embeds_neg],dim=0)
            vl_output = self.itm_head(vl_embeddings)

            itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(bs,dtype=torch.long)],
                                dim=0).to(self.device)
        

        itm_labels = itm_labels.to(self.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)  

        _, pred = vl_output.max(1)
        hits = pred == itm_labels

        if stage=='dev':
            return loss_lm, loss_ita, loss_itm, gen_sentence
        else:
            return loss_lm, loss_ita, loss_itm, hits
    
    def generate(self, source_ids):
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)        
        preds = []   
        zero = torch.cuda.LongTensor(1).fill_(0) 

        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                     for y in encoder_output.past_key_values]
            beam = Beam(self.beam_size,self.sos_id,self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
            for _ in range(self.max_length): 
                if beam.done():
                    break
                # input_ids = input_ids.to(self.device)
                ids = torch.cat((context_ids,input_ids),-1)
                mask = self.bias[:,context_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
                mask = mask & ids[:,None,:].ne(1)
                out = self.decoder(input_ids,attention_mask=mask,past_key_values=context).last_hidden_state
                hidden_states = out[:,-1,:]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids,beam.getCurrentState()),-1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))

        preds = torch.cat(preds,0)    

        return preds   
        
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient   
                param.requires_grad = True

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, code_feat, text_feat):
        # gather keys before updating queue
        code_feats = concat_all_gather(code_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = code_feats.shape[0]

        ptr = int(self.queue_ptr)
    
        assert self.queue_size % code_feats.shape[0] == 0  # for simplicity
        self.code_queue[:, ptr:ptr + batch_size] = code_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr 

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     

# class Beam(object):
#     def __init__(self, size,sos,eos, device='cuda'):
#         self.size = size
#         print(device)
#         if device == 'cuda':
#             self.tt = torch.cuda
#         else:
#             self.tt = torch
#         self.device=device
#         # The score for each translation on the beam.
#         self.scores = self.tt.FloatTensor(size).zero_()
#         # The backpointers at each time-step.
#         self.prevKs = []
#         # The outputs at each time-step.
#         self.nextYs = [self.tt.LongTensor(size)
#                        .fill_(0)]
#         self.nextYs[0][0] = sos
#         # Has EOS topped the beam yet.
#         self._eos = eos
#         self.eosTop = False
#         # Time and k pair for finished.
#         self.finished = []

#     def getCurrentState(self):
#         "Get the outputs for the current timestep."
#         batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
#         batch = batch.to(self.device)
#         return batch

#     def getCurrentOrigin(self):
#         "Get the backpointers for the current timestep."
#         return self.prevKs[-1]

#     def advance(self, wordLk):
#         """
#         Given prob over words for every last beam `wordLk` and attention
#         `attnOut`: Compute and update the beam search.
#         Parameters:
#         * `wordLk`- probs of advancing from the last step (K x words)
#         * `attnOut`- attention at the last step
#         Returns: True if beam search is complete.
#         """
#         numWords = wordLk.size(1)

#         # Sum the previous scores.
#         if len(self.prevKs) > 0:
#             beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

#             # Don't let EOS have children.
#             for i in range(self.nextYs[-1].size(0)):
#                 if self.nextYs[-1][i] == self._eos:
#                     beamLk[i] = -1e20
#         else:
#             beamLk = wordLk[0]
#         flatBeamLk = beamLk.view(-1)
#         bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

#         self.scores = bestScores

#         # bestScoresId is flattened beam x word array, so calculate which
#         # word and beam each score came from
#         prevK = bestScoresId // numWords
#         self.prevKs.append(prevK)
#         self.nextYs.append((bestScoresId - prevK * numWords))


#         for i in range(self.nextYs[-1].size(0)):
#             if self.nextYs[-1][i] == self._eos:
#                 s = self.scores[i]
#                 self.finished.append((s, len(self.nextYs) - 1, i))

#         # End condition is when top-of-beam is EOS and no global score.
#         if self.nextYs[-1][0] == self._eos:
#             self.eosTop = True

#     def done(self):
#         return self.eosTop and len(self.finished) >= self.size

#     def getFinal(self):
#         if len(self.finished) == 0:
#             self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
#         self.finished.sort(key=lambda a: -a[0])
#         if len(self.finished) != self.size:
#             unfinished=[]
#             for i in range(self.nextYs[-1].size(0)):
#                 if self.nextYs[-1][i] != self._eos:
#                     s = self.scores[i]
#                     unfinished.append((s, len(self.nextYs) - 1, i)) 
#             unfinished.sort(key=lambda a: -a[0])
#             self.finished+=unfinished[:self.size-len(self.finished)]
#         return self.finished[:self.size]

#     def getHyp(self, beam_res):
#         """
#         Walk back to construct the full hypothesis.
#         """
#         hyps=[]
#         for _,timestep, k in beam_res:
#             hyp = []
#             for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
#                 hyp.append(self.nextYs[j+1][k])
#                 k = self.prevKs[j][k]
#             hyps.append(hyp[::-1])
#         return hyps
    
#     def buildTargetTokens(self, preds):
#         sentence=[]
#         for pred in preds:
#             tokens = []
#             for tok in pred:
#                 if tok==self._eos:
#                     break
#                 tokens.append(tok)
#             sentence.append(tokens)
#         return sentence
        
class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
        