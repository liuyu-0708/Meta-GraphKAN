
from xml.etree.ElementTree import canonicalize
from rdkit import Chem

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import numpy as np
import pandas as pd

import os
from itertools import repeat, product, chain

from .utils import _load_tox21_dataset, _load_sider_dataset
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)


from structuralremap_construction.data.bondgraph.to_bondgraph import to_bondgraph
from structuralremap_construction.data.featurization.mol_to_data import mol_to_data
from structuralremap_construction.data.featurization.smiles_to_3d_mol import smiles_to_3d_mol

allowable_features = {
    # 描述了可能的原子序数列表，从1到118。这些数字对应于元素周期表中的元素。
    'possible_atomic_num_list':
    list(range(1, 119)),
    # 列出了可能的形式电荷，从-5到+5。形式电荷是指原子在分子中所带的电荷，可以是正的、负的或零。
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    # 描述了可能的手性类型，包括未指定的手性（CHI_UNSPECIFIED）、四面体碳原子的手性（CHI_TETRAHEDRAL_CW 和 CHI_TETRAHEDRAL_CCW，
    # 分别表示顺时针和逆时针）、以及其他类型的手性（CHI_OTHER）。
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    # 列出了可能的杂化类型，包括S（sp杂化）、SP（sp杂化）、SP2、SP3、SP3D、SP3D2和未指定（UNSPECIFIED）。
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    # 描述了原子可能结合的氢原子数量，从0到8。
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    # 列出了可能的隐式价，从0到6。隐式价是指原子在分子中可能形成的化学键的数量。
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    # 描述了原子可能的度数，即与原子相连的键的数量，从0到10。
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # 列出了可能的键类型，包括单键（SINGLE）、双键（DOUBLE）、三键（TRIPLE）和芳香键（AROMATIC）。
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
    ],
    # 描述了双键的立体化学方向，包括无方向（NONE）、向上（ENDUPRIGHT）和向下（ENDDOWNRIGHT）。这些特征用于表示双键的立体化学特性。
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # 定义原子特征的数量，这里只有原子类型和手性标签两种特征
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    #  遍历分子中的每个原子
    for atom in mol.GetAtoms():
        #  获取原子的特征，包括原子序数和手性标签，并将其转换为索引
        atom_feature = [
            allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())
        ] + [
            allowable_features['possible_chirality_list'].index(
                atom.GetChiralTag())
        ]
        atom_features_list.append(atom_feature)
    # 将原子特征列表转换为 PyTorch 张量
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds# 定义键特征的数量，这里只有键类型和键方向两种特征
    num_bond_features = 2  # bond type, bond direction
    # 检查分子是否有键
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        # 获取键的特征，包括键类型和键方向，并将其转换为索引
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                allowable_features['possible_bonds'].index(bond.GetBondType())
            ] + [
                allowable_features['possible_bond_dirs'].index(
                    bond.GetBondDir())
            ]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            # 由于无向图，需要添加反向边
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # 将边的连接列表转换为 PyTorch 张量
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # 将边的特征列表转换为 PyTorch 张量
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                    dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    # 创建图数据对象，包含原子特征、边连接和边特征
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, SMILES = Chem.MolToSmiles(mol, canonical = True ))

    return data


class MoleculeDataset(InMemoryDataset):
    """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
    def __init__(self,
                 root,
                 dataset,
                 
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 empty=False):


        self.root = root
        self.dataset = dataset
        
        self.task_counts = [0, 0]  

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)

        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)

        return file_name_list 

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt' 

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')  
    


    def process(self):
        data_smiles_list = []
        data_list = []
        task_counts = [0,0]  

        if self.dataset == "tox21" or self.dataset == "muv":
            smiles_list, rdkit_mol_objs, labels = _load_tox21_dataset(self.raw_paths[0])

            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_data(rdkit_mol)
                if data.edge_index.shape[1] == 0:
                    print(f"WARNING: Skipping molecule {i} because it has no edges.")
                    continue
                data = to_bondgraph(data)
                data.pos = None
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i, :])

                data_list.append(data)
                data_smiles_list.append(smiles_list[i])


                label = data.y.item()
                if label == 0:
                    task_counts[0] += 1  
                else:
                    task_counts[1] += 1  

        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = _load_sider_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_data(rdkit_mol)
                if data.edge_index.shape[1] == 0:
                    print(f"WARNING: Skipping molecule {i} because it has no edges.")
                    continue
                data = to_bondgraph(data)
                data.pos = None
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])


                label = data.y.item()
                if label == 0:
                    task_counts[0] += 1 
                else:
                    task_counts[1] += 1  
    

        else:
            raise ValueError('Invalid dataset name')

        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir, 'smiles.csv'), index=False, header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        self.update_accumulated_counts(task_counts)

        print('Task counts:', task_counts)
        print("Total accumulated task counts:", self.task_counts)

    def update_accumulated_counts(self, task_counts=None):
        if task_counts is not None:
            self.task_counts[0] += task_counts[0]
            self.task_counts[1] += task_counts[1]
        print("Updated accumulated task counts:", self.task_counts)

    def get_task_counts(self):
        return self.task_counts

    def display_accumulated_counts(self):
        print("Total accumulated task counts:", self.task_counts)




    def get(self, idx):
        res_data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            if isinstance(item, torch.Tensor):
                s = list(repeat(slice(None), item.dim()))
                s[res_data.__cat_dim__(key,
                                       item)] = slice(slices[idx],
                                                      slices[idx + 1])
            else:
                s = slices[idx].item()
            res_data[key] = item[s]
        return res_data


if __name__ == "__main__":
    task_counts_list = []  
    
    for i in range(1,18):
        root = "/root/codes/MolFeSCue-master-2/fsadmet/data/muv/new/{}".format(i)
        dataset = MoleculeDataset(root, dataset="muv")
        task_counts = dataset.get_task_counts()
        task_counts_list.append(task_counts)

    print("Task counts for each dataset:", task_counts_list)

    formatted_task_counts = [[int(num) for num in counts] for counts in task_counts_list]
    print(formatted_task_counts)

