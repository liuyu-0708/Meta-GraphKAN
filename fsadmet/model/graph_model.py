
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

from .layer import GINConv, GCNConv, GraphSAGEConv, GATConv
from torch_geometric.nn import SAGPooling, TopKPooling

class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self,
                 num_layer,
                 emb_dim,
                 JK="last",
                 drop_ratio=0.5,
                 num_atom_type=120,
                 num_chirality_tag=3,
                 gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        if self.JK == "concat":
            self.jk_proj = torch.nn.Linear((self.num_layer + 1) * emb_dim,
                                           emb_dim)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        self.topk_pools = torch.nn.ModuleList()  
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):

            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:

                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h),
                              self.drop_ratio,
                              training=self.training)
            
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_rep = torch.cat(h_list, dim=1)
            # node_rep = 0.6*h_list[0] + 0.3*h_list[1] + 0.1*h_list[2]
            node_rep = self.jk_proj(node_rep)
        elif self.JK == "last":
            node_rep = h_list[-1]

        elif self.JK == "max":
            h_list = [torch.unsqueeze(h, 0) for h in h_list]
            node_rep = torch.max(torch.cat(h_list, dim=0),
                                 keepdim=False,
                                 dim=0)[0]
        elif self.JK == "sum":
            h_list = [torch.unsqueeze(h, 0) for h in h_list]
            node_rep = torch.sum(torch.cat(h_list, dim=0),
                                 keepdim=False,
                                 dim=0)
        return node_rep


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self,
                 num_layer,
                 emb_dim,
                 num_tasks,
                 JK="last",
                 drop_ratio=0,
                 graph_pooling="mean",
                 gnn_type="gin",
                 pooling_ratio = 0.8 ):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.num_workers = 2
        self.pooling_ratio = pooling_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim,
                                    set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim,
                                                 self.num_tasks)

    def from_pretrained(self, model_file):



        pretrained_state_dict = torch.load(model_file, map_location='cpu')


        model_state_dict = self.gnn.state_dict()
        pretrained_filtered = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}

        missing_keys = set(model_state_dict.keys()) - set(pretrained_filtered.keys())
        if missing_keys:
            print(f'Warning: Missing keys in state_dict: {missing_keys}')


        unexpected_keys = set(pretrained_state_dict.keys()) - set(model_state_dict.keys())
        if unexpected_keys:
            print(f'Warning: Unexpected keys in state_dict: {unexpected_keys}')

        self.gnn.load_state_dict(pretrained_filtered, strict=False)



    def forward(self, *argv):


        if len(argv) >= 4:

            data, x, edge_index, edge_attr, batch, x1, edge_index1, batch1, edge_attr1 = argv
        elif len(argv) == 1:

            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")
        node_rep = self.gnn(x, edge_index, edge_attr)
        node_rep1 = self.gnn(x1, edge_index1, edge_attr1)

        graph_rep = self.pool(node_rep, batch)
        graph_rep1 = self.pool(node_rep1, batch1)
        pred = self.graph_pred_linear(graph_rep)
        pred1 = self.graph_pred_linear(graph_rep1)


        return pred, graph_rep, node_rep, pred1, graph_rep1, node_rep1

   


if __name__ == "__main__":
    pass
