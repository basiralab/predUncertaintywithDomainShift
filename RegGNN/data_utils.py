'''
Utility functions for data processing.
'''

import copy
from scipy.sparse import coo_matrix
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
import torch_geometric
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from config import Config


def simulate_dataset(conn_mean=0.0, conn_std=1.0,
                     score_mean=90.0, score_std=10.0,
                     n_subjects=30, seed=None):
    rng = np.random.default_rng(seed)
    conn = rng.normal(loc=conn_mean, scale=conn_std, size=(Config.ROI, Config.ROI, n_subjects))
    score = rng.normal(loc=score_mean, scale=score_std, size=(n_subjects,1))
    return torch.tensor(conn), torch.tensor(score)




def load_dataset_pytorch(connectome,score):
    '''Loads the data for the given population into a list of Pytorch Geometric
    Data objects, which then can be used to create DataLoaders.
    '''
    

    connectomes = torch.load(f"{Config.DATA_FOLDER}{connectome}")
    scores = torch.load(f"{Config.DATA_FOLDER}{score}")

    # n_subjects = scores.size()[0]
    # for subject in range(n_subjects):
    #     connectomes[:, :, subject] = torch.from_numpy(MinMaxScaler().fit_transform(connectomes[:, :, subject]))
        
    
    connectomes[connectomes < 0] = 0


    pyg_data = []
    for subject in range(scores.shape[0]):
        
        sparse_mat = to_sparse(connectomes[:, :, subject])
        pyg_data.append(torch_geometric.data.Data(x=torch.eye(Config.ROI, dtype=torch.float),
                                                  y=scores[subject].float(), edge_index=sparse_mat._indices(),
                                                  edge_attr=sparse_mat._values().float()))
        

    return pyg_data


def to_sparse(mat):
    '''Transforms a square matrix to torch.sparse tensor

    Methods ._indices() and ._values() can be used to access to
    edge_index and edge_attr while generating Data objects
    '''
    coo = coo_matrix(mat, dtype='float64')
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    coo_index = torch.stack([row, col], dim=0)
    coo_values = torch.from_numpy(coo.data.astype(np.float64).reshape(-1, 1)).reshape(-1)
    sparse_mat = torch.sparse.LongTensor(coo_index, coo_values)
    return sparse_mat



def get_loaders(train, test, batch_size=1):
    '''Returns data loaders for given data lists
    '''
    
    train_loader = torch_geometric.data.DataLoader(train, batch_size=batch_size)
    test_loader = torch_geometric.data.DataLoader(test, batch_size=batch_size)
    return train_loader, test_loader


def load_dataset_tensor(connectome="connectome.ts",score="score.ts",pop="NT",only_positive=True):
    '''Loads dataset as tuple of (tensor of connectomes,
       tensor of fiq scores, tensor of viq scores)
    '''
    connectomes = torch.load(f"{Config.DATA_FOLDER}{connectome}")
    scores = torch.load(f"{Config.DATA_FOLDER}{score}")
    if only_positive:
        connectomes[connectomes < 0] = 0
        

    return connectomes, scores


def to_dense(data):
    '''Returns a copy of the data object in Dense form.
    '''
    denser = torch_geometric.transforms.ToDense()
    copy_data = denser(copy.deepcopy(data))
    return copy_data
