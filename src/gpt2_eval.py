# This code is modified from https://bastings.github.io/annotated_encoder_decoder/
import argparse
import logging
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
import csv
import sys

from modeling_gpt2 import GPT2LMHeadModel
from tokenization_gpt2 import GPT2Tokenizer
from rct_utils import MEDDataset

from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu

import warnings

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove )
        logits[indices_to_remove] = filter_value
    return logits

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_sequence_nucleus(model, length, start_token=None, batch_size=None, context=None, temperature=0.8, top_k=0, top_p=0.9, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    
    with torch.no_grad():
        for i in range(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            log_probs = F.softmax(filtered_logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
                if prev.item() == bos:
                    break
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
                if prev.item() == bos:
                    break

            output = torch.cat((output, prev), dim=1)
    return output


def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=0.7, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in range(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
                if prev.item() == bos:
                    break
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
                if prev.item() == bos:
                    break

            output = torch.cat((output, prev), dim=1)
    return output

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='../models/gpt2_h_1_1.pt', help='pretrained model name or path to local checkpoint')
parser.add_argument('--dev_file', type=str, default='../data/dev_seq2seq_small.csv', help='validation data file name')
parser.add_argument('--pred_file', type=str, default='pred_test.csv', help='output prediction file name')
parser.add_argument('--example_num', type=int, default=200, help='output example number, set to `-1` to run all examples')
args = parser.parse_args()
print(args)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

enc = GPT2Tokenizer.from_pretrained("gpt2")
bos = enc.encoder['<|endoftext|>']

dev_set = MEDDataset(args.dev_file, enc)

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.load_state_dict(torch.load(args.model_name))
model.cuda()

if __name__ == '__main__':
    model.eval()
    with torch.no_grad():
        hyps = []
        refs = []
        save_pred = []
        # test the first 200 examples
        for i in range(args.example_num if args.example_num != -1 else len(dev_set)):
            (q, a), toks = dev_set.get_plain(i)
            sent = sample_sequence_nucleus(model, length=100, batch_size=1, context=toks, sample=True)
            out = sent[0, len(toks):].tolist()
            text = enc.decode(out)
            print("Example[%d]"%(i))
            print("SRC:", q)
            print("-")
            print("TRG:", a)
            print("-")
            try:
                print("GEN:", text)
            except:
                print("GEN: `Skip this example. Possible error occurred in decoding text since gpt2 generated irrigular coding.`")
            print("-"*50, flush=True)
            refs.append(a.lower())
            hyps.append(text.lower())
            save_pred.append([' '.join(q.split(' ')[-6:]), text, a])
        
        hyps = [(x if x != "" else "<|endoftext|>") for x in hyps]
        # rouge of 200 samples
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)
        print("ROUGE-1 : ", scores['rouge-1'])
        print("ROUGE-2 : ", scores['rouge-2'])
        print("ROUGE-L : ", scores['rouge-l'])

        # bleu of 200 samples
        warnings.simplefilter("ignore")
        score = corpus_bleu([[ref.split(' ')] for ref in refs], [hyp.split(' ') for hyp in hyps])
        print("BLEU : ", score)

        # save prediction to file
        with open(args.pred_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(save_pred)

        sys.stdout.flush()


