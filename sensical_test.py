import torch
import torch.nn as nn
from torch.autograd import Variable
import data
from utils import batchify, get_batch, repackage_hidden
from scipy.stats import pointbiserialr as pb
from scipy.stats import kendalltau as kt
import math

import itertools

# checkpoint = './checkpoints/WT2.pt'
# checkpoints = '../models/BNC.18hr.QRNN.pt'
data = 'data/bnc'
torch.cuda.set_device(1)
device = torch.device(1)

torch.manual_seed(1234)


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

sent = 'colourless green ideas sleep furiously'.split()
sent_color = 'colorless green ideas sleep furiously'.split()

def get_total_prob(model, perm):
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
    return total_prob.item()

model = torch.load('../models/BNC_QRNN_CHECKPOINTS/checkpoint_ep6.pt', map_location=device)[0]
g = lambda x: get_total_prob(model, x.split())

gt = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, None, 1, None, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, None, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]

x = []
y_exp = []
y_log = []

for i, perm in enumerate(itertools.permutations(sent)):

    if gt[i] == 1:
        x.append(1)
        score = get_total_prob(model, perm)
        y_log.append(score)
        y_exp.append(math.exp(score))

f = open('bnc_grammatical.txt').read().split('\n')
for ff in f:
    try:
        score = get_total_prob(model, ff.strip().lower().split())
        x.append(0)
        y_log.append(score)
        y_exp.append(math.exp(score))
    except:
        print('error occur!')
# for j, perm in enumerate(itertools.permutations(sent_color)):
# 
#     if x[j] is not None:
#         y_color.append(math.exp(get_total_prob(model, perm)))


print('LOG', pb(x, y_log), kt(x, y_log))
print('EXP', pb(x, y_exp), kt(x, y_exp))
