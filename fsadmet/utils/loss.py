

import os
import sys
from nt_xent import NT_Xent
import torch.distributed as dist
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import torch.nn as nn
from .train_utils import build_negative_edges, build_negative_edges1
from pytorch_metric_learning import miners, losses

from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import numpy as np


class HardPairMiner(BaseMiner):

    def __init__(self, alpha_s, alpha_e, beta, **kwargs):
        super().__init__(**kwargs)
        self.alpha_s = alpha_s
        self.alpha_e = alpha_e
        self.beta = beta
        self.f_t = lambda step: self.alpha_s * np.exp(-beta * step
                                                      ) + self.alpha_e

    def mine(self, embeddings, labels, t, ref_emb=None, ref_labels=None):

        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)

        neg_pairs = mat[a2, n]
        alpha = self.f_t(t)
        pos_mask = torch.arange(0, len(a1) - 1, dtype=torch.long)
        neg_mask = torch.argsort(
            neg_pairs, descending=False)[:int(len(neg_pairs) * alpha)]

        return a1[pos_mask], p[pos_mask], a2[neg_mask], n[neg_mask]


class LossGraphFunc():

    def __init__(self, selfsupervised_weight, contrastive_weight, alpha_s,
                 alpha_e, beta):
        self.selfsupervised_weight = selfsupervised_weight
        self.contrastive_weight = contrastive_weight
        self.criterion = nn.BCEWithLogitsLoss()

        if contrastive_weight > 0:
            self.miner = HardPairMiner(alpha_s=alpha_s,
                                       alpha_e=alpha_e,
                                       beta=beta)
            self.contrastive = losses.ContrastiveLoss(pos_margin=0,
                                                      neg_margin=1)

        if self.selfsupervised_weight > 0:
            self.bond_criterion = nn.BCEWithLogitsLoss()
            self.mask_criterion = nn.CrossEntropyLoss()

    def __call__(self, meta_model, batch, node_emb, graph_emb, pred, step, node_emb1,
                              graph_emb1,pred1,):

        self_atom_loss = torch.tensor(0.0)
        self_bond_loss = torch.tensor(0.0)
        contrastive_loss = torch.tensor(0.0) 

        self_atom_loss1 = torch.tensor(0.0)
        self_bond_loss1 = torch.tensor(0.0)



        y = batch.y.view(pred.shape).to(torch.float64)

        loss = torch.sum(self.criterion(pred.double(), y)) / pred.size()[0]

        if self.contrastive_weight > 0:
            hard_pairs = self.miner.mine(graph_emb, y.squeeze(), step)
            contrastive_loss = self.contrastive(graph_emb, y.squeeze(),
                                                hard_pairs)
            loss += (self.contrastive_weight * contrastive_loss)



        # selfsupervised loss
        if self.selfsupervised_weight > 0:

            # edge reconstruction loss
            positive_score = torch.sum(node_emb[batch.edge_index[0, ::2]] *
                                       node_emb[batch.edge_index[1, ::2]],
                                       dim=1)

            negative_edge_index = build_negative_edges(batch)
            negative_score = torch.sum(node_emb[negative_edge_index[0]] *
                                       node_emb[negative_edge_index[1]],
                                       dim=1)

            self_bond_loss = torch.sum(
                self.bond_criterion(positive_score,
                                    torch.ones_like(positive_score)) + self.
                bond_criterion(negative_score, torch.zeros_like(
                    negative_score))) / negative_edge_index[0].size()[0]

            ## add bond loss to total loss
            loss += (self.selfsupervised_weight * self_bond_loss)

            # atom prediction loss
            mask_num = random.sample(range(0, node_emb.size()[0]), y.shape[0])

            pred_emb = meta_model.masking_linear(node_emb[mask_num])

            self_atom_loss = self.mask_criterion(pred_emb, batch.x[mask_num,
                                                                   0])

            ## add atom loss to total loss
            loss += (self.selfsupervised_weight * self_atom_loss)


            mask_num = random.sample(range(0, node_emb1.size()[0]), y.shape[0])

            pred_emb = meta_model.masking_linear(node_emb1[mask_num])

            self_atom_loss1 = self.mask_criterion(pred_emb, batch.x1[mask_num,
                                                                   0])

            ## add atom loss to total loss
            loss += (self.selfsupervised_weight * self_atom_loss1)

        return loss, self_atom_loss, self_bond_loss, contrastive_loss



class LossSeqFunc():

    def __init__(self, contrastive_weight, alpha_s, alpha_e, beta):
        self.contrastive_weight = contrastive_weight
        self.alpha_s = alpha_s
        self.alpha_e = alpha_e
        self.beta = beta
        self.f_t = lambda step: self.alpha_s * np.exp(-beta * step
                                                      ) + self.alpha_e

        self.criterion = nn.BCEWithLogitsLoss()

        if contrastive_weight > 0:
            self.miner = HardPairMiner(alpha_s=alpha_s,
                                       alpha_e=alpha_e,
                                       beta=beta)
            self.contrastive = losses.ContrastiveLoss(pos_margin=0,
                                                      neg_margin=1)

    def __call__(self, y, pred, seq_emb, step):

        contrastive_loss = torch.tensor(0)
        loss = torch.sum(self.criterion(pred, y.float())) / pred.size()[0]

        if self.contrastive_weight > 0:
            hard_pairs = self.miner.mine(seq_emb, y.squeeze(), step)
            contrastive_loss = self.contrastive(seq_emb, y.squeeze(),
                                                hard_pairs)
            loss += (self.contrastive_weight * contrastive_loss)

        return loss, contrastive_loss
