
import os  

from omegaconf import DictConfig  
import torch 

from fsadmet.model.graph_model import GNN_graphpred  
from utils.std_logger import Logger  
from fsadmet.model.meta_model import MetaGraphModel, MetaSeqModel  
from transformers import AutoModel, AutoTokenizer  

from fsadmet.model.models.model import Graph

def model_preperation(orig_cwd: str, cfg: DictConfig, args) -> torch.nn.Module:
    
    if cfg.model.backbone == "gnn":

        base_learner = Graph(args,num_features_xd=128,dropout=0.2,aug_ratio=0.4,weights=[0.4, 0.4, 0.2, 0.6, 0.3, 0.1])

        model = MetaGraphModel(base_learner,  
                               cfg.meta.selfsupervised_weight,  
                               cfg.model.gnn.emb_dim)  
    Logger.info("模型加载成功！......\n")
    Logger.info(model)

    return model


