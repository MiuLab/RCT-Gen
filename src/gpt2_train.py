# This code is modified from https://bastings.github.io/annotated_encoder_decoder/
import argparse
import logging
import math
import copy
import time
import csv
import sys

from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu

from modeling_gpt2 import GPT2LMHeadModel
from tokenization_gpt2 import GPT2Tokenizer
from rct_utils import MEDDataset, collate

import warnings

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
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
parser.add_argument('--save_model_name', type=str, default='gpt2_model', help='pretrained model name or path to local checkpoint')
parser.add_argument('--train_file', type=str, default='../data/train_seq2seq_small.csv', help='training data file name')
parser.add_argument('--dev_file', type=str, default='../data/dev_seq2seq_small.csv', help='validation data file name')
parser.add_argument("--n_epochs", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument('--pred_file', type=str, default='pred_test', help='output prediction file name')
parser.add_argument('--example_num', type=int, default=200, help='output example number, set to `-1` to run all examples')
parser.add_argument("--mode", type=str, default="train")
args = parser.parse_args()
print(args)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

enc = GPT2Tokenizer.from_pretrained("gpt2")
bos = enc.encoder['<|endoftext|>']

train_set = MEDDataset(args.train_file, enc)
dev_set = MEDDataset(args.dev_file, enc)

train_iter = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
dev_iter = torch.utils.data.DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)


model = GPT2LMHeadModel.from_pretrained("gpt2")
model.cuda()

critirion = nn.CrossEntropyLoss(ignore_index=-1)
opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

def train_batch(model, batch, backward=True):
    (sent, rang, label) = batch
    output, _ = model(sent.cuda(), causal_range=rang)
    loss = critirion(output.transpose(-1, -2), label.cuda())
    if backward:
        loss.backward()
    return float(loss.cpu().float())


if __name__ == "__main__":
    best_ppl = 1000.0
    for epoch in range(args.n_epochs+1):
        model.train()
        start = time.time()
        logger.info("*** Start Training Epoch [%d/%d] ***\n" % (epoch, args.n_epochs))
        my_iter = iter(train_iter)
        if epoch > 0:
            for step in range(len(train_iter)):
                try:
                    batch = my_iter.next()
                except:
                    print("batch failed")
                    continue
                if batch[0].shape[-1] > 500:
                    continue
                opt.zero_grad()
                loss = train_batch(model, batch)
                opt.step()
                now = time.time()
                rem = ((len(train_iter) - step) / max(step, 1)) * max((now - start), 1)
                if(step % 500 == 0):
                    print("Epoch [{epoch}/{n_epoch}] Step[{step}/{n_step}] ETA:{eta:.1f} Loss: {loss:.4f} Perplexity: {ppl:.4f}".format(
                        epoch=epoch,
                        n_epoch=args.n_epochs,
                        step=step,
                        n_step=len(train_iter),
                        eta=rem,
                        loss=loss,
                        ppl=float(math.exp(loss))),
                        flush=True
                    )
        logger.info("*** Start Testing Epoch [%d/%d] ***\n" % (epoch, args.n_epochs))
        
        model.eval()
        with torch.no_grad():
            print("="*50)
            print("="*18, "Epoch [%d/%d]"%(epoch, args.n_epochs), "="*18)
            print("=" * 50)

            hyps = []
            refs = []
            save_pred = []
            for i in range(args.example_num if args.example_num != -1 else len(dev_set)):
                (q, a), toks = dev_set.get_plain(i)
                sent = sample_sequence(model, length=100, batch_size=1, context=toks, sample=False)
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

            # prevent error occurred from gpt2 generated empty sequence
            hyps = [(x if x != "" else "<|endoftext|>") for x in hyps]

            # rouge of 200 samples
            rouge = Rouge()
            scores = rouge.get_scores(hyps, refs, avg=True)
            print("ROUGE : ", scores)

            # bleu of 200 samples
            warnings.simplefilter("ignore")
            score = corpus_bleu([[ref.split(' ')] for ref in refs], [hyp.split(' ') for hyp in hyps])
            print("BLEU : ", score)

            # save prediction to file
            with open(args.pred_file+'_{}.csv'.format(epoch), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerows(save_pred)


            dev_loss = 0.0
            dev_num = 0.0
            for step, batch in enumerate(dev_iter):
                dev_loss += train_batch(model, batch, backward=False)
                dev_num += 1
            dev_perplexity = math.exp(dev_loss/dev_num)
            logger.info("*** Valid Perplexity: %.4f ***\n" % dev_perplexity)

            val_ppl = dev_perplexity
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                print('best_ppl: {:.4f}'.format(best_ppl), flush=True)
                torch.save(model.state_dict(), args.save_model_name+".ep_{}.pt".format(epoch))
            
            sys.stdout.flush()


