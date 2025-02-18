import json
from rdkit.Chem import AllChem
import numpy as np
from rdkit import Chem

def my_collate_fn(batch):
    all_smiles  = [_[0] for _ in batch]
    all_y =  [_[1] for _ in batch]
    # all_y = np.concatenate(all_y, axis=0 )
    return all_smiles, all_y

def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """

    with open(input_path) as json_file:
        binary_list = json.load(json_file)

    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)

    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]


    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 

    return smiles_list, rdkit_mol_objs_list, labels


def _load_sider_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """

    with open(input_path) as json_file:
        binary_list = json.load(json_file)

    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    rdkit_mol_objs_list = [Chem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1

    num_zeros = np.sum(labels == 0)
    num_ones = np.sum(labels == 1)

    return smiles_list, rdkit_mol_objs_list, labels

if __name__ == "__main__":
    input_path = "/home/richard/projects/fsadmet/data/tox21/new/12/raw/tox21.json"
    _load_tox21_dataset(input_path)


