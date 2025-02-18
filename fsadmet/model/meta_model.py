
import os

import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer


class attention(nn.Module):

    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(nn.Linear(dim, 100), nn.ReLU(),
                                    nn.Linear(100, 1))
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.softmax_layer(torch.transpose(x, 1, 0))
        return x


class Interact_attention(nn.Module):

    def __init__(self, dim, num_tasks):
        super(Interact_attention, self).__init__()
        self.layers = nn.Sequential(nn.Linear(num_tasks * dim, dim), nn.Tanh())

    def forward(self, x):
        x = self.layers(x)
        return x


class MetaGraphModel(nn.Module):

    def __init__(self, base_model, selfsupervised_weight, emb_dim):
        super(MetaGraphModel, self).__init__()
        self.base_model = base_model
        self.emb_dim = emb_dim
        self.selfsupervised_weight = selfsupervised_weight

        if self.selfsupervised_weight > 0:
            self.masking_linear = nn.Linear(self.emb_dim, 119)


class MetaSeqModel(nn.Module):

    def __init__(self, base_model, tokenizer):
        super(MetaSeqModel, self).__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 768))
        self.final_layer = nn.Linear(768, 1)

    def forward(self, x):


        pred_y = self.base_model(**x)[0]


        pool_output = self.global_avg_pool(pred_y).squeeze(1)


        pred = self.final_layer(pool_output)


        return pred, pool_output
