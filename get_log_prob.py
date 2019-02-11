import torch
import torch.nn as nn
from torch.autograd import Variable
import data
from utils import batchify, get_batch, repackage_hidden

import itertools

# checkpoint = './checkpoints/WT2.pt'
checkpoint = 'language_models/wt2_qrnn.pt'
data = 'data/wikitext-2'
torch.cuda.set_device(2)
device = torch.device(2)

with open(checkpoint, 'rb') as f:
    model, criterion, _ = torch.load(f, map_location=device)


import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(data.encode()).hexdigest())
if os.path.exists(fn):
    print('loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('producing dataset...')
    corpus = data.corpus(args.data)
    torch.save(corpus, fn)



dictionary = corpus.dictionary

def tokenize_sent(sent):
    return torch.LongTensor([dictionary.word2idx[x] for x in sent]).cuda()

sent = 'colorless green ideas dream furiously'.split()

for perm in itertools.permutations(sent):

    tok_sent = tokenize_sent(perm).unsqueeze(1)

    model.eval()
    model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    # data, targets = get_batch(data_source, i, args, evaluation=True)
    data = tok_sent
    output, hidden = model(data, hidden)
    output = model.decoder(output)

    log_softmaxed = torch.log_softmax(output, 1)
    log_probs = torch.gather(log_softmaxed, 1, data)
    total_prob = torch.sum(log_probs)

    print(' '.join(perm), total_prob.item())
