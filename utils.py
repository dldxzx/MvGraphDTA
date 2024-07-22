import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        self.data_len = len(data)
        self.id = data['PDBID']
        # self.smile = data['Smiles']
        # self.protein = data['Sequence']
        # self.pocket = data['Pocket']
        self.affinity = torch.from_numpy(np.array(data['affinity']).astype(np.float32))
    def __getitem__(self, index):
        return self.id[index], self.affinity[index]
        # return self.id[index], self.smile[index], self.protein[index], self.pocket[index], self.affinity[index]
    def __len__(self):
        return self.data_len

class dataset_for_other(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        self.data_len = len(data)
        self.cid = data['PubChem_CID']
        self.pid = data['Uniprot_ID']
        self.affinity = torch.from_numpy(np.array(data['Affinity']).astype(np.float32))
    def __getitem__(self, index):
        return self.cid[index],self.pid[index], self.affinity[index]
    def __len__(self):
        return self.data_len
    

class dataset_for_interaction(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        self.data_len = len(data)
        self.cid = data['cid']
        self.pid = data['pid']
        self.interaction = torch.from_numpy(np.array(data['interaction']).astype(np.int64))

    def __getitem__(self, index):
        return self.cid[index], self.pid[index], self.interaction[index]
    
    def __len__(self):
        return self.data_len