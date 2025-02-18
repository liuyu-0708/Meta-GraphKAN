from typing import List

from torch_geometric.data import Data, Batch

from structuralremap_construction.data.bondgraph.to_bondgraph import to_bondgraph
from structuralremap_construction.data.featurization.mol_to_data import mol_to_data
from structuralremap_construction.data.featurization.smiles_to_3d_mol import smiles_to_3d_mol


import copy



def smiles_to_data(smiles: str) -> Data:
    mol = smiles_to_3d_mol(smiles)  # 从 SMILES 创建 3D 分子结构
    data = mol_to_data(mol)  # 将分子结构转换为 Data 对象
    data = to_bondgraph(data)  # 转换为 bond graph 形式
    data.pos = None  # 清除位置信息
    return data


# def collate_with_circle_index(data_list: List[Data]) -> Batch:
#     """
#     Collates a list of Data objects into a Batch object.

#     Args:
#         data_list: a list of Data objects. Each Data object must contain `circle_index` attribute.
#         k_neighbors: number of k consecutive neighbors to be used in the message passing step.

#     Returns:
#         Batch object containing the collate attributes from data objects, including `circle_index` collated
#         to shape (total_num_nodes, max_circle_size).

#     """
#     batch1 = copy.deepcopy(data_list)  # 创建一个深拷贝
#     print(f'type of batch_1:', type(batch1))
#     print(f'type of data_list:', type(data_list))
#     for b in batch1:
#         b.x = b.x1  # 处理 x 属性

#     batch1 = Batch.from_data_list(batch1, exclude_keys=['circle_index'])  # 创建第一个 Batch 对象
#     batch = Batch.from_data_list(data_list, exclude_keys=['circle_index'])  # 创建第二个 Batch 对象
#     batch.batch1 = batch1.batch  # 复制 batch 属性
#     batch.edge_index1 = batch1.edge_index1  # 复制 edge_index1 属性
#     return batch


from typing import List
from copy import deepcopy
from torch_geometric.data import Data, Batch

# def collate_with_circle_index(data_list: Batch) -> Batch:
#     """
#     Collates a list of Data objects into a Batch object.

#     Args:
#         data_list: a list of Data objects. Each Data object must contain `circle_index` attribute.
#         k_neighbors: number of k consecutive neighbors to be used in the message passing step.

#     Returns:
#         Batch object containing the collate attributes from data objects, including `circle_index` collated
#         to shape (total_num_nodes, max_circle_size).
#     """

#     # 打印类型检查结果
#     print(f'type of data_list:', type(data_list))
    
#     # 创建一个深拷贝
#     batch1 = deepcopy(data_list)

#     print(f'type of batch1: ', type(batch1))
    
#     print(f'before transform :', batch1)
#     batch1.x = batch1.x1
#     # # 处理 x 属性
#     # for b in batch1:
#     #     b.x = b.x1  # 处理 x 属性
    
#     # 创建第一个 Batch 对象
#     batch1 = Batch.from_data_list(batch1, exclude_keys=['circle_index'])
    
#     # 创建第二个 Batch 对象
#     batch = Batch.from_data_list(data_list, exclude_keys=['circle_index'])
    
#     # 复制 batch 属性
#     batch.batch1 = batch1.batch
    
#     # 复制 edge_index1 属性
#     batch.edge_index1 = batch1.edge_index1
    
#     print(f'after transform :', batch)

#     return batch


from typing import List
from copy import deepcopy
from torch_geometric.data import Batch

def collate_with_circle_index(data_batch: Batch) -> Batch:
    """
    Collates a Batch object into a new Batch object with updated attributes.

    Args:
        data_batch: A Batch object that contains multiple Data objects. Each Data object must contain `circle_index` attribute.
        k_neighbors: number of k consecutive neighbors to be used in the message passing step.

    Returns:
        Batch object containing the collate attributes from data objects, including `circle_index` collated
        to shape (total_num_nodes, max_circle_size).
    """
    # 确认 data_batch 是 Batch 类型
    if not isinstance(data_batch, Batch):
        raise ValueError("data_batch must be a Batch object.")
    
    # 获取数据列表
    data_list = data_batch.to_data_list()
    
    # 打印类型检查结果
    # print(f'type of data_list:', type(data_list))
    
    # 创建一个深拷贝
    batch1 = deepcopy(data_list)
    
    # 打印类型检查结果
    # print(f'type of batch1 before transformation:', type(batch1))
    
    # 处理 x 属性
    for b in batch1:
        # if not hasattr(b, 'x1'):
            # 如果没有 x1 属性，则添加一个默认的 x1 属性
            # b.x1 = b.x.clone()  # 默认使用 x 作为 x1
            # 根据实际情况初始化 x1
            # b.x1 = torch.zeros_like(b.x)  # 初始化为零张量
        
        # 将 x1 赋值给 x
        b.x = b.x1
        # b.edge_attr = b.edge_attr1
    
    # 创建第一个 Batch 对象
    try:
        batch1 = Batch.from_data_list(batch1, exclude_keys=['circle_index'])
        print(f'batch1: ', batch1)
    except Exception as e:
        print(f"Error creating batch1: {e}")
        raise
    
    # 打印类型检查结果
    # print(f'type of batch1 after transformation:', type(batch1))
    
    # 创建第二个 Batch 对象
    batch = Batch.from_data_list(data_list, exclude_keys=['circle_index'])
    print(f'batch: ', batch)

    
    # 复制 batch 属性
    batch.batch1 = batch1.batch
    
    # 复制 edge_index1 属性
    
    batch.edge_index1 = batch1.edge_index1
    
    # 打印最终结果
    # print(f'after transform :', batch)

    return batch


def collate_with_circle_index2(data_list: List[Data]) -> Batch:
    """
    Collates a list of Data objects into a Batch object.

    Args:
        data_list: a list of Data objects. Each Data object must contain `circle_index` attribute.
        k_neighbors: number of k consecutive neighbors to be used in the message passing step.

    Returns:
        Batch object containing the collate attributes from data objects, including `circle_index` collated
        to shape (total_num_nodes, max_circle_size).

    """
    
    batch1 = copy.deepcopy(data_list)  
    for b in batch1:

        b.x = b.x1  
        # print(b)

    batch1 = Batch.from_data_list(batch1, exclude_keys=['circle_index'])  
    batch = Batch.from_data_list(data_list, exclude_keys=['circle_index'])  


    batch.batch1 = batch1.batch  
    batch.edge_index1 = batch1.edge_index1  
    # print(f'transform batch1: ', batch1)
    # print(f'transform batch: ', batch)
    return batch