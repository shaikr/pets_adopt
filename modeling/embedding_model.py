import torch
from torch import nn
from torch.nn.init import kaiming_normal
from torch.nn import functional as F

def emb_init(x):
    x = x.weight.data
    sc = 2 / (x.size(1) + 1)
    x.uniform_(-sc, sc)


class EmbeddingModel(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops, y_range=None, use_bn=False, classify=None):
        super().__init__()  ## inherit from nn.Module parent class
        self.embs = nn.ModuleList([nn.Embedding(m, d) for m, d in emb_szs])  ## construct embeddings
        for emb in self.embs: emb_init(emb)  ## initialize embedding weights
        n_emb = sum(e.embedding_dim for e in self.embs)  ## get embedding dimension needed for 1st layer
        szs = [n_emb + n_cont] + szs  ## add input layer to szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i + 1]) for i in
            range(len(szs) - 1)])  ## create linear layers input, l1 -> l1, l2 ...
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]])  ## batchnormalization for hidden layers activations
        for o in self.lins: kaiming_normal(o.weight.data)  ## init weights with kaiming normalization
        # self.outp = nn.Linear(szs[-1], out_sz)  ## create linear from last hidden layer to output
        # kaiming_normal(self.outp.weight.data)  ## do kaiming initialization

        self.emb_drop = nn.Dropout(emb_drop)  ## embedding dropout, will zero out weights of embeddings
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])  ## fc layer dropout
        self.bn = nn.BatchNorm1d(n_cont)  # bacthnorm for continous data
        self.use_bn, self.y_range = use_bn, y_range
        self.classify = classify

    def forward(self, x_cat, x_cont):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]  # takes necessary emb vectors
        x = torch.cat(x, 1)  ## concatenate along axis = 1 (columns - side by side) # this is our input from cats
        x = self.emb_drop(x)  ## apply dropout to elements of embedding tensor
        x2 = self.bn(x_cont)  ## apply batchnorm to continous variables
        x = torch.cat([x, x2], 1)  ## concatenate cats and conts for final input
        for l, d, b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))  ## dotprod + non-linearity
            if self.use_bn: x = b(x)  ## apply batchnorm activations
            x = d(x)  ## apply dropout to activations
        # x = self.outp(x)  # we defined this externally just not to apply dropout to output
        # if self.classify:
        #     x = F.sigmoid(x)  # for classification
        # elif y_range:
        #     x = F.sigmoid(x)  ## scales the output between 0,1
        #     x = x * (self.y_range[1] - self.y_range[0])  ## scale output
        #     x = x + self.y_range[0]  ## shift output
        return x
