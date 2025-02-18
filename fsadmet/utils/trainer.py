
import os
import sys
# import nni
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import numpy as np
from structuralremap_construction import *
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import  Data
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as DataLoaderChem
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
import torch.optim as optim
import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from fsadmet.dataset.dataset import MoleculeDataset
from fsadmet.dataset.dataset_chem import MoleculeDataset as MoleculeDatasetChem
from fsadmet.dataset.utils import my_collate_fn
from fsadmet.model.samples import sample_datasets, sample_test_datasets
from .train_utils import update_params
from sklearn.metrics import f1_score, average_precision_score
from .loss import LossGraphFunc, LossSeqFunc
from .std_logger import Logger
from .FocalLoss import BCEFocalLoss
from concurrent.futures import ThreadPoolExecutor

import nni

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from torch.nn.functional import relu, softmax, elu
from ana.tsne_analysis import TSNEModule,PCAModule

from nt_xent import NT_Xent

class Trainer(object):

    def __init__(self, meta_model, cfg, device):

        self.meta_model = meta_model 
        self.device = device
        self.cfg = cfg
        self.dataset_name = cfg.data.dataset
        self.data_path_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            cfg.data.path)

        self.num_tasks = cfg.tasks[self.dataset_name].num_tasks
        self.num_train_tasks = len(cfg.tasks[self.dataset_name].train_tasks)
        self.num_test_tasks = len(cfg.tasks[self.dataset_name].test_tasks)
        self.n_way = cfg.tasks[self.dataset_name].n_way
        self.m_support = cfg.tasks[self.dataset_name].m_support
        self.k_query = cfg.tasks[self.dataset_name].k_query

        self.batch_size = cfg.train.batch_size
        self.meta_lr = cfg.train.meta_lr
        self.update_lr = cfg.train.update_lr
        self.update_step = cfg.train.update_step
        self.update_step_test = cfg.train.update_step_test
        self.eval_epoch = cfg.train.eval_epoch

        self.saved_model_metric = 0
        self.default_metric = 0
        self.default_f1 = 0
        self.default_pr_auc = 0
        if cfg.model.backbone == "gnn":

            self.optimizer = optim.Adam(
                self.meta_model.base_model.parameters(),
                lr=cfg.train.meta_lr,
                weight_decay=cfg.train.decay)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        # # formal---gama---4
        # self.criterion = BCEFocalLoss(gamma=4, alpha=0.25, reduction='elementwise_mean')
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.update_lr)
        self.epochs = cfg.train.epochs
        self.num_workers = cfg.data.num_workers

    def train_epoch(self, epoch):

        # samples dataloaders
        support_loaders = []
        query_loaders = []

        self.meta_model.base_model.train()

        for task in self.cfg.tasks[self.dataset_name].train_tasks:
            # for task in tasks_list:

            if self.cfg.model.backbone == "gnn":
                dataset = MoleculeDataset(os.path.join(self.data_path_root,
                                                       self.dataset_name,
                                                       "new", str(task+1)),
                                          dataset=self.dataset_name)


                collate_fn = None
                MyDataLoader = DataLoader

            support_dataset, query_dataset = sample_datasets(
                dataset, self.dataset_name, task, self.n_way, self.m_support,
                self.k_query)

            support_loader = MyDataLoader(support_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=self.num_workers,
                                          collate_fn=collate_fn,
                                          )

            query_loader = MyDataLoader(query_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_workers,
                                        collate_fn=collate_fn,
                                       )
            

            
            support_loaders.append(support_loader)
            query_loaders.append(query_loader)

        for k in range(0, self.update_step):

            old_params = parameters_to_vector(
                self.meta_model.base_model.parameters())

            # use this loss to save all the losses of query set
            losses_q = torch.tensor([0.0]).to(self.device)


            # losses_q = []

            for task in range(self.num_train_tasks):

                losses_s = torch.tensor([0.0]).to(self.device)

                # training support
                for _, batch in enumerate(
                        tqdm(
                            support_loaders[task],
                            desc=
                            "Training | Epoch: {} | UpdateK: {} | Task: {} | Support Iteration"
                            .format(epoch, k, task + 1))):

                    if self.cfg.model.backbone == "gnn":
                        lossz = 0
                        batch = batch.to(self.device) 

                        batch.x = batch.x.to(torch.long)
                        batch.edge_index = batch.edge_index.to(torch.long)
                        batch.edge_attr = batch.edge_attr.to(torch.long)
                        batch.x1 = batch.x1.to(torch.long)
                        batch.edge_index1 = batch.edge_index1.to(torch.long)
                        batch.edge_attr1 = batch.edge_attr1.to(torch.long)
                        data = collate_with_circle_index(batch)
                        data.edge_attr = None
                        batch1 = data.batch1.detach()
                        edge = data.edge_index1.detach()
                        xd = data.x1.detach()
                        # self.optimizer.zero_grad()
                        output, x_g, x_g1, output1 = self.meta_model.base_model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)

                        loss_1 = self.criterion(output, data.y.view(-1, 1).float())

                        criterion1 = NT_Xent(output.shape[0], 0.1, 1)
                        criterion2=NT_Xent(output.shape[0], 0.1, 1)

                        loss_2 = self.criterion(output1, data.y.view(-1, 1).float())

                        cl_loss_node = criterion1(x_g, x_g1)
                        cl_loss_graph=criterion2(output,output1)
                        # loss = loss_2

                        loss = loss_1+loss_2+(0.1*cl_loss_node)+(0.1*cl_loss_graph)

                        losses_s += loss


                if not self.cfg.mode.nni and self.cfg.logger.log:
                    mlflow.log_metric(
                        f"Training/support_task_{task + 1}_loss",
                        losses_s.item(),
                        step=epoch * self.update_step + k
                    )                

                _, new_params = update_params(self.meta_model.base_model,
                                                losses_s,
                                                update_lr=self.update_lr)
                

                vector_to_parameters(new_params,
                                        self.meta_model.base_model.parameters())
                # print('Train epoch: {} \tLoss: {:.6f}'.format(epoch, lossz))


                this_loss_q = torch.tensor([0.0]).to(self.device)


                # training query task set
                for _, batch in enumerate(
                        tqdm(
                            query_loaders[task],
                            desc=
                            "Training | Epoch: {} | UpdateK: {} | Task: {} | Query Iteration"
                            .format(epoch, k, task + 1))):

                    if self.cfg.model.backbone == "gnn":
                        batch = batch.to(self.device) 

                        batch.x = batch.x.to(torch.long)
                        batch.edge_index = batch.edge_index.to(torch.long)
                        batch.edge_attr = batch.edge_attr.to(torch.long)
                        batch.x1 = batch.x1.to(torch.long)
                        batch.edge_index1 = batch.edge_index1.to(torch.long)
                        batch.edge_attr1 = batch.edge_attr1.to(torch.long)
                        data = collate_with_circle_index(batch)
                        data.edge_attr = None
                        batch1 = data.batch1.detach()
                        edge = data.edge_index1.detach()
                        xd = data.x1.detach()
                        # self.optimizer.zero_grad()
                        output, x_g, x_g1, output1 = self.meta_model.base_model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)

                        loss_1 = self.criterion(output, data.y.view(-1, 1).float())

                        criterion1 = NT_Xent(output.shape[0], 0.1, 1)
                        criterion2=NT_Xent(output.shape[0], 0.1, 1)

                        loss_2 = self.criterion(output1, data.y.view(-1, 1).float())

                        cl_loss_node = criterion1(x_g, x_g1)
                        cl_loss_graph=criterion2(output,output1)

                        loss = loss_1+loss_2+(0.1*cl_loss_node)+(0.1*cl_loss_graph)

                        this_loss_q += loss


                if not self.cfg.mode.nni and self.cfg.logger.log:
                    mlflow.log_metric(
                        f"Training/query_task_{task + 1}_loss",
                        this_loss_q.item(),
                        step=epoch * self.update_step + k)


                if task == 0:
                    losses_q = this_loss_q
                else:
                    losses_q = torch.cat((losses_q, this_loss_q), 0)

                vector_to_parameters(old_params, self.meta_model.base_model.parameters())


            loss_q = torch.sum(losses_q) / self.num_train_tasks


            if not self.cfg.mode.nni and self.cfg.logger.log:
                mlflow.log_metric(
                    "Training/weighted_query_loss",
                    loss_q.item(),
                    step=epoch * self.update_step + k)


            self.optimizer.zero_grad()

            loss_q.backward()

            self.optimizer.step()

            return []

    def test(self, epoch):

        accs = []
        rocs = []
        f1_scores = []
        pr_aucs = []


        all_features = []  # Characteristics for storing all tasks
        all_labels = []    # Real labels for storing all tasks
        old_params = parameters_to_vector(
            self.meta_model.base_model.parameters())

        for task in self.cfg.tasks[self.dataset_name].test_tasks:

            if self.cfg.model.backbone == "gnn":
                dataset = MoleculeDataset(os.path.join(self.data_path_root,
                                                       self.dataset_name,
                                                       "new", str(task+1)),
                                          dataset=self.dataset_name)
                collate_fn = None
                MyDataLoader = DataLoader


            support_dataset, query_dataset = sample_test_datasets(
                dataset, self.dataset_name, task,
                self.n_way, self.m_support, self.k_query)

            support_loader = MyDataLoader(support_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=self.num_workers,
                                          collate_fn=collate_fn,
                                          )
            query_loader = MyDataLoader(query_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_workers,
                                        collate_fn=collate_fn,
                                        )

            self.meta_model.eval()

            for k in range(0, self.update_step_test):
                lossz = torch.tensor([0.0]).to(self.device)


                for step, batch in enumerate(
                        tqdm(
                            support_loader,
                            desc=
                            "Testing | Epoch: {} | UpdateK: {} | Task: {} | Support Iteration"
                            .format(epoch, k, task))):

                    if self.cfg.model.backbone == "gnn":
                        
                        batch = batch.to(self.device) 

                        batch.x = batch.x.to(torch.long)
                        batch.edge_index = batch.edge_index.to(torch.long)
                        batch.edge_attr = batch.edge_attr.to(torch.long)
                        batch.x1 = batch.x1.to(torch.long)
                        batch.edge_index1 = batch.edge_index1.to(torch.long)
                        batch.edge_attr1 = batch.edge_attr1.to(torch.long)
                        data = collate_with_circle_index(batch)
                        data.edge_attr = None
                        batch1 = data.batch1.detach()
                        edge = data.edge_index1.detach()
                        xd = data.x1.detach()
                        # self.optimizer.zero_grad()
                        output, x_g, x_g1, output1 = self.meta_model.base_model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)

                        loss_1 = self.criterion(output, data.y.view(-1, 1).float())

                        criterion1 = NT_Xent(output.shape[0], 0.1, 1)
                        criterion2=NT_Xent(output.shape[0], 0.1, 1)

                        loss_2 = self.criterion(output1, data.y.view(-1, 1).float())

                        cl_loss_node = criterion1(x_g, x_g1)
                        cl_loss_graph=criterion2(output,output1)


                        loss = loss_1+loss_2+(0.1*cl_loss_node)+(0.1*cl_loss_graph)

                        lossz += loss

                if not self.cfg.mode.nni and self.cfg.logger.log:
                    mlflow.log_metric(
                        "Testing/support_task_{}_loss".format(task),
                        lossz.item(),
                        step=epoch * self.update_step_test + k)
                new_grad, new_params = update_params(
                    self.meta_model.base_model, lossz, update_lr=self.update_lr)

                vector_to_parameters(new_params,
                                     self.meta_model.base_model.parameters())

            y_true = []
            y_scores = []
            y_predict = []


            for _, batch in enumerate(
                    tqdm(
                        query_loader,
                        desc=
                        "Testing | Epoch: {} | UpdateK: {} | Task: {} | Query Iteration"
                        .format(epoch, k, task))):

                if self.cfg.model.backbone == "gnn":
                    batch = batch.to(self.device)
                    with torch.no_grad():

                        batch.x = batch.x.to(torch.long)
                        batch.edge_index = batch.edge_index.to(torch.long)
                        batch.edge_attr = batch.edge_attr.to(torch.long)
                        batch.x1 = batch.x1.to(torch.long)
                        batch.edge_index1 = batch.edge_index1.to(torch.long)
                        batch.edge_attr1 = batch.edge_attr1.to(torch.long)
                        data = collate_with_circle_index(batch)
                        data.edge_attr = None
                        batch1 = data.batch1.detach()
                        edge = data.edge_index1.detach()
                        xd = data.x1.detach()

                        output, x_g, x_g1, output1 = self.meta_model.base_model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)
                        # features = torch.cat([x_g, x_g1], dim=1).cpu().numpy()  # 合并两个特征表示
                        features = x_g1.cpu().numpy()
                        labels = batch.y.cpu().numpy()
                        all_features.append(features)
                        all_labels.append(labels)

                        y_score = torch.sigmoid(output.squeeze()).cpu()
                        y_scores.append(y_score)
                        y_predict.append(
                            torch.tensor([1 if _ > 0.5 else 0 for _ in y_score],
                                            dtype=torch.long).cpu())

                        y_true.append(batch.y.cpu())



            y_true = torch.cat(y_true, dim=0).numpy()
            y_scores = torch.cat(y_scores, dim=0).numpy()
            y_predict = torch.cat(y_predict, dim=0).numpy()


            # print(f"y_scores: {y_scores}")
            # print(f"y_true: {y_true}")

            unique_classes = set(y_true)
            if len(unique_classes) == 1:
                print("Only one class present in y_true. ROC AUC score is not defined in that case.")
                roc_score = 0.5  
            else:
                roc_score = roc_auc_score(y_true, y_scores)
            acc_score = accuracy_score(y_true, y_predict)


            f1 = f1_score(y_true, y_predict)
            pr_auc = average_precision_score(y_true, y_scores)

            if not self.cfg.mode.nni and self.cfg.logger.log:

                mlflow.log_metric("Testing/query_task_{}_auc".format(task),
                                    roc_score,
                                    step=epoch)
            accs.append(acc_score)
            rocs.append(roc_score)
            f1_scores.append(f1)
            pr_aucs.append(pr_auc)
        if not self.cfg.mode.nni and self.cfg.logger.log:

            mlflow.log_metric("Testing/query_mean_auc",
                              np.mean(rocs),
                              step=epoch)

        vector_to_parameters(old_params,
                            self.meta_model.base_model.parameters())



        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        
        
        return accs, rocs,f1_scores, pr_aucs, all_features, all_labels



    def run(self):

        # tsne_module = TSNEModule(output_dir='/root/codes/MolFeSCue-master-2/tsne_visualizations/')
        pca_module = PCAModule(output_dir='/root/codes/MolFeSCue-master-2/pca_visualizations/')


        for epoch in range(1, self.epochs + 1):

            self.train_epoch(epoch)

            # torch.cuda.empty_cache()
            if epoch % self.eval_epoch == 0:
                accs, rocs,f1_scores, pr_aucs,all_features, all_labels = self.test(epoch)

                mean_roc = round(np.mean(rocs), 4)
                mean_f1 = round(np.mean(f1_scores), 4)
                mean_auc = round(np.mean(pr_aucs), 4)

                if mean_roc > self.default_metric:
                    self.default_metric = mean_roc
                    self.default_f1 = mean_f1
                    self.default_pr_auc = mean_auc

                # Logger.info("downstream task accs: {}".format(
                #     [round(_, 3) for _ in accs]))
                Logger.info("downstream task aucs: {}".format(
                    [round(_, 4) for _ in rocs]))
                Logger.info(
                    "mean downstream task mean auc、f1、pr_auc: {},{},{}".format(mean_roc, mean_f1, mean_auc))
                

                if mean_roc > self.saved_model_metric:  
                    torch.save(self.meta_model.base_model.state_dict(), '/root/codes/MolFeSCue-master-2/model_output/model_epoch_{}.pth'.format(epoch))
                    self.saved_model_metric = mean_roc  




                if self.cfg.mode.nni:
                    nni.report_intermediate_result({"default_auc": np.mean(rocs)})


                
                # tsne_module.visualize(all_features, all_labels, epoch=epoch, 
                #                     title=f't-SNE visualization of the graph embeddings (Epoch {epoch})',
                #                     filename=f't-SNE_visualization_epoch_{epoch}.png')
                pca_module.visualize(all_features, all_labels, epoch=epoch, 
                                    title=f'PCA visualization of the graph embeddings (Epoch {epoch})',
                                    filename=f'PCA_visualization_epoch_{epoch}.png')

        nni.report_final_result({
        "default_auc": self.default_metric,
        "f1": self.default_f1,
        "pr_auc": self.default_pr_auc
        })
  
