""" Medical dataset loader with gpt2 BPE tokenization."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import random
from io import open
import time
import random
import csv

from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset



def collate(li):
    max_len = max([x[1][1] for x in li])
    toks = []
    rngs = []
    labs = []
    for (sent, rg, label) in li:
        toks.append(sent + [0] * (max_len - rg[1]))
        rngs.append(rg)
        labs.append(label + [-1] * (max_len - rg[1]))
    return torch.tensor(toks), rngs, torch.tensor(labs)

class MEDDataset(Dataset):
    """ Loading MED RCT data and processing with BPE for gpt-2 training. """

    def __init__(self, corpus_path, enc, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.enc = enc
        self.bos = enc.encoder['<|endoftext|>']
        if not os.path.exists(self.corpus_path):
            raise IOError("corpus_path %s does not exist"%self.corpus_path)
        # load samples into memory
        if self.corpus_lines is None:
            self.corpus_lines = int(os.popen("wc -l %s" % corpus_path).read().split(' ')[0])

        self.corpus_entries = list(csv.reader(open(corpus_path, "r", encoding=encoding)))


    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.corpus_lines

    def __getitem__(self, item):
        if isinstance(item, slice):
            #Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*item.indices(len(self)))]

        q, a = self.corpus_entries[item]
        org_token_q = self.enc.encode(q)
        token_q = [self.bos] + org_token_q
        token_a = self.enc.encode(a)
        label = [-1] * len(org_token_q) + token_a + [self.bos]
        f = len(token_q)
        t = f + len(token_a)
        return (token_q + token_a), [f, t], label
    
    def get_plain(self, item):
        if isinstance(item, slice):
            #Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*item.indices(len(self)))]

        q, a = self.corpus_entries[item]
        org_token_q = self.enc.encode(q)
        token_q = [self.bos] + org_token_q
        # token_a = self.enc.encode(a)
        return (q, a), (token_q)
