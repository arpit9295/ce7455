import torch
from scipy.stats import spearmanr
import os

import data


device = torch.device("cpu")

corpus = data.Corpus('./data/wikitext-2')

with open('./model.pt', 'rb') as f:
    model = torch.load(f).to(device)

path = os.path.join('./wordsim353_sim_rel/', 'wordsim_similarity_goldstandard.txt')

w1_indexes = []
w2_indexes = []
targets = []
predictions = []
pvalues = []

with open(path, 'r', encoding="utf8") as f:
    for line in f:
        w1, w2, correlation = line.split()
        if w1 in corpus.dictionary.word2idx and w2 in corpus.dictionary.word2idx:
            w1_indexes.append(corpus.dictionary.word2idx[w1])
            w2_indexes.append(corpus.dictionary.word2idx[w2])
            targets.append(float(correlation))
        else:
            print("words", w1, w2, "not in vocabulary")

w1_indexes = torch.tensor(w1_indexes, requires_grad=False).to(device)
w2_indexes = torch.tensor(w2_indexes, requires_grad=False).to(device)
targets = torch.tensor(targets, requires_grad=False).to(device)

w1_emb = model.encoder(w1_indexes).detach()
w2_emb = model.encoder(w2_indexes).detach()

cos = torch.nn.CosineSimilarity(dim=1)
similarity = cos(w1_emb, w2_emb)

correlation, pvalue = spearmanr(similarity, targets, axis=0)

with open('correlation.txt', 'w') as outf:
    outf.write('{:.4f}'.format(correlation))
    outf.write('\tSpearman Correlation\n')


    outf.write('\n\n======================================\n\n')

    for i in range(0, len(similarity)):
        outf.write('{:.2f}'.format(similarity[i]))
        outf.write('\t')
        outf.write('{:.2f}'.format(targets[i]))
        outf.write('\t')
        outf.write(corpus.dictionary.idx2word[w1_indexes[i]])
        outf.write('\t')
        outf.write(corpus.dictionary.idx2word[w2_indexes[i]])
        outf.write('\n')
