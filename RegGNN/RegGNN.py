'''RegGNN regression model architecture.

torch_geometric needs to be installed.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseGCNConv


class RegGNN(nn.Module):
    '''Regression using a DenseGCNConv layer from pytorch geometric.

       Layers in this model are identical to GCNConv.   
    '''

    def __init__(self, nfeat, nhid, nclass,hid_layer_num,linear_hid_layer_num, dropout):
        super(RegGNN, self).__init__()

        self.gc1 = DenseGCNConv(nfeat, nhid)
        self.gcs = nn.ModuleList([DenseGCNConv(nhid, nhid) for i in range(hid_layer_num)])
        self.gc2 = DenseGCNConv(nhid, nclass)
        self.dropout = dropout
        self.linears = nn.ModuleList([nn.Linear(nfeat, nfeat) for i in range(linear_hid_layer_num)])
        self.LinearLayer = nn.Linear(nfeat, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        for i, l in enumerate(self.gcs):
            x = self.gcs[i](x,edge_index) + l(x,edge_index)
        x = self.gc2(x, edge_index)
        x = torch.transpose(x, 2, 1)
        for i, l in enumerate(self.linears):
            x = self.linears[i](x) + l(x)
        x = self.LinearLayer(x)
        
        return x

    def loss(self, pred, score):
        return F.mse_loss(pred, score)
        
    
    #return self.gauss_loss(pred, score)
    
    # def gauss_loss(self,y_pred,y_true):
    #     """Negative log likelihood of y_true, with the likelihood defined by a normal distribution."""
    #     y_true = y_true[0]
    #     means = y_pred[0]
    #     # We predict the log of the standard deviation, so exponentiate the prediction here
        
        
    #     stds = torch.exp(y_pred[1])
    #     variances = stds * stds
        
    #     torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
        
    #     log_p = (-torch.log(torch.sqrt(2 * torch.pi * variances))
    #              -(y_true - means)*(y_true - means)/(2 * variances))

    #     return -log_p