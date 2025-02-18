from collections import defaultdict

import torch
import torch_geometric
import torch_geometric.data
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected



def to_bondgraph(data: Data) -> Data:
    

    if not is_undirected(data.edge_index):
        edge_index, edge_attr = to_undirected(edge_index=data.edge_index, edge_attr=data.edge_attr)
    else:
        edge_index, edge_attr = data.edge_index, data.edge_attr

    if data.edge_index is None or data.edge_attr is None:
        raise ValueError("Edge index and edge attributes must be provided.")
    if data.edge_index.size(1) != data.edge_attr.size(0):
        raise ValueError("The number of edges does not match the number of edge attributes.")
    new_nodes = []
    new_nodes_to_idx = {}
    for edge, edge_attr in zip(edge_index.T, edge_attr):
        a, b = edge
        a, b = a.item(), b.item()
        a2b = torch.cat([data.x[a], edge_attr, data.x[b]])  # x_{i, j} = x'_i | e'_{i, j} | x'_j.
        new_nodes_to_idx[(a, b)] = len(new_nodes)
        new_nodes.append(
            {'a': a, 'b': b, 'a_attr': data.x[a], 'node_attr': a2b, 'old_edge_attr': edge_attr,})
        
    if len(new_nodes) == 0:
        raise ValueError("No new nodes were created, possibly due to an empty edge_index or incorrect edge processing.")

    in_nodes = defaultdict(list)
    for i, node_dict in enumerate(new_nodes):
        a, b = node_dict['a'], node_dict['b']
        in_nodes[b].append({'node_idx': i, 'start_node_idx': a})
    new_edges = []
    # Create incoming node information.
    for i, node_dict in enumerate(new_nodes):
        a, b = node_dict['a'], node_dict['b']
        ab_old_edge_attr = node_dict['old_edge_attr']
        a_attr = node_dict['a_attr']
        a_in_nodes_indices = [d['node_idx'] for d in in_nodes[a]]
        # Concatenate features for new edge representation.
        for in_node_c in a_in_nodes_indices:
            in_node = new_nodes[in_node_c]
            ca_old_edge_attr = in_node['old_edge_attr']
            # e_{(i, j), (j, k)} = e'_(i, j) | x'_j | e'_{k, j}:
            edge_attr = torch.cat([ca_old_edge_attr, a_attr, ab_old_edge_attr])
            new_edges.append({'edge': [in_node_c, i], 'edge_attr': edge_attr})

    # parallel_node_index = []
    # for node_dict in new_nodes:
    #     a, b = node_dict['a'], node_dict['b']
    #     parallel_idx = new_nodes_to_idx[(b, a)]
    #     parallel_node_index.append(parallel_idx)
# Prepare new graph structure with transformed nodes and edges

    if len(new_edges) == 0:
        raise ValueError("No new edges were created, possibly due to an empty edge_index or incorrect edge processing.")
    new_x = [d['node_attr'] for d in new_nodes]
    new_edge_index = [d['edge'] for d in new_edges]
    new_edge_attr = [d['edge_attr'] for d in new_edges]
    new_x = torch.stack(new_x)
    new_edge_index = torch.tensor(new_edge_index).T
    new_edge_attr = torch.stack(new_edge_attr)
    # parallel_node_index = torch.tensor(parallel_node_index)
# Create a new Data object for the transformed graph.

    data = torch_geometric.data.Data(x=new_x, x1=data.x,edge_index1=data.edge_index,edge_index=new_edge_index, edge_attr=new_edge_attr, edge_attr1 = data.edge_attr)

    return data
