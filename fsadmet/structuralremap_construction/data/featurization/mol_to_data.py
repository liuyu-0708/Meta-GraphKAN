
import numpy as np
import rdkit
import torch
from rdkit.Chem import rdMolTransforms, Mol, Atom, Bond
from torch_geometric.data import Data
from typing import List
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
atom_types = ['H', 'C', 'B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
formal_charges = [-1, -2, 1, 2, 0]
degree = [0, 1, 2, 3, 4, 5, 6]
num_hs = [0, 1, 2, 3, 4]
local_chiral_tags = [0, 1, 2, 3]
hybridization = [
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
]
bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_data(mol: Mol) -> Data:
    """
    Transforms a rdkit mol object into a torch_geometric Data object.
    Args:
        mol: rdKit mol object.

    Returns:
        Data object containing the following attributes:
            - x: node features.
            - edge_index: edge index.
            - edge_attr: edge features.
            - pos: node positions.
    """
    # Edge Index
    # Edge Index
    # 检查是否包含氢原子，这里的if(1):始终为真，意味着总是使用RDKit的GetAdjacencyMatrix方法来获取邻接矩阵
    
    if(1):#1 means it contains H
        adj = rdkit.Chem.GetAdjacencyMatrix(mol)
        # 并将其转换为无向边索引
        edge_index = adjacency_to_undirected_edge_index(adj)
    else:

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])
        edges_array = np.array(edge_index)

        # 转换为edge_index形式
        edge_index = np.transpose(edges_array)

    # Edge Features
    # Edge Features
    # 根据边索引获取分子中的键，并生成边特征
    bonds = []
    for b in range(int(edge_index.shape[1] / 2)):
        bond_index = edge_index[:, ::2][:, b]
        bond = mol.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = get_edge_features(bonds)

    # Node Features
    # 获取分子中的所有原子，并为每个原子生成节点特征
    atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
    atom_features_list = []

    for atom in mol.GetAtoms():
        atom_feature = [
            allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum()),  # 原子编号
            allowable_features['possible_chirality_list'].index(atom.GetChiralTag()),   # 手性标记
            allowable_features['possible_formal_charge_list'].index(atom.GetFormalCharge()),  # 形式电荷
            allowable_features['possible_hybridization_list'].index(atom.GetHybridization()),  # 杂化类型
            allowable_features['possible_numH_list'].index(atom.GetTotalNumHs()),  # 氢原子数
            allowable_features['possible_implicit_valence_list'].index(atom.GetImplicitValence()),  # 隐式价
            allowable_features['possible_degree_list'].index(atom.GetDegree())  # 原子度
        ]
        # 将原子特征列表转换为PyTorch张量
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    ########
    # 调用get_node_features函数获取节点特征
    node_features = get_node_features(atoms)
    
    # Positions
    # torch.as_tensor(node_features)
    # Create Data object
    # 创建PyTorch Geometric的Data对象，包含节点特征、边索引、边特征
    data = Data(x=torch.as_tensor(node_features),
                edge_index=torch.as_tensor(edge_index).long(),
                edge_attr=torch.as_tensor(edge_features),
                )

    return data

def one_hot_embedding(value: int, options: List[int]) -> List[int]:
    """
    Encodes a value into a one-hot embedding.
    Args:
        value: a value which index will be retrieved from options and encoded.
        options: a list of possible values.

    Returns:
        One-hot embedding of the value.
    """
    embedding = [0] * (len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding


def adjacency_to_undirected_edge_index(adj: np.ndarray) -> np.ndarray:
    """
    Converts an adjacency matrix into an edge index.
    Args:
        adj: adjacency matrix.

    Returns:
        Edge index.
    """
    adj = np.triu(np.array(adj, dtype=int))  
    array_adj = np.array(np.nonzero(adj), dtype=int)  
    edge_index = np.zeros((2, 2 * array_adj.shape[1]), dtype=int)  
    edge_index[:, 1::2] = np.flipud(array_adj)
    return edge_index



def get_node_features(atoms: List[Atom]) -> np.ndarray:
    """
    Gets an array of node features from a list of atoms.
    Args:
        atoms: list of atoms of shape (N).

    Returns:
        Array of node features of shape (N, 43).
    """
    num_features = (len(atom_types) + 1) + \
                   (len(degree) + 1) + \
                   (len(formal_charges) + 1) + \
                   (len(num_hs) + 1) + \
                   (len(hybridization) + 1) + \
                   2  # 43
    node_features = np.zeros((len(atoms), num_features))
    for node_index, node in enumerate(atoms):
        features = one_hot_embedding(node.GetSymbol(), atom_types)  # atom symbol, dim=12 + 1
        features += one_hot_embedding(node.GetTotalDegree(), degree)  # total number of bonds, H included, dim=7 + 1
        features += one_hot_embedding(node.GetFormalCharge(), formal_charges)  # formal charge, dim=5+1
        features += one_hot_embedding(node.GetTotalNumHs(), num_hs)  # total number of bonded hydrogens, dim=5 + 1
        features += one_hot_embedding(node.GetHybridization(), hybridization)  # hybridization state, dim=7 + 1
        features += [int(node.GetIsAromatic())]  # whether atom is part of aromatic system, dim = 1
        features += [node.GetMass() * 0.01]  # atomic mass / 100, dim=1
        node_features[node_index, :] = features
    return np.array(node_features, dtype=np.float32)


def get_edge_features(bonds: List[Bond]) -> np.ndarray:
    """
    Gets an array of edge features from a list of bonds.
    Args:
        bonds: a list of bonds of shape (N).

    Returns:
        Array of edge features of shape (N, 7).
    """
    num_features = (len(bond_types) + 1) + 2  # 7
    edge_features = np.zeros((len(bonds) * 2, num_features))
    for edge_index, edge in enumerate(bonds):
        features = one_hot_embedding(str(edge.GetBondType()), bond_types)  
        features += [int(edge.GetIsConjugated())]  
        features += [int(edge.IsInRing())]  

        edge_features[2 * edge_index: 2 * edge_index + 2, :] = features
    return np.array(edge_features, dtype=np.float32)


def get_positions(mol: rdkit.Chem.Mol) -> np.ndarray:
    """
    Gets the 3D positions of the atoms in the molecule.
    Args:
        mol: a molecule embedded in 3D space with N atoms.

    Returns:
        Array of positions of shape (N, 3).
    """
    conf = mol.GetConformer()
    return np.array(
        [
            [
                conf.GetAtomPosition(k).x,
                conf.GetAtomPosition(k).y,
                conf.GetAtomPosition(k).z,
            ]
            for k in range(mol.GetNumAtoms())
        ]
    )
