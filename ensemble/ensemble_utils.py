
import numpy as np
import torch

from config import Config
from RegGNN.data_utils import to_dense
from PNA.PNA.model import compute_deg


def e_u(preds,uncs,score,epsilon=1e-6):
    """
        Proposed metric calculation according to model outputs for uncertanity and target score prediction.
        outputs error value.

        Parameters
        ----------
        preds: target output score prediction array for each subject in cohort.
        uncs: Predicted uncertanity array for each subject in cohort.
        score: ground truth array for each subject in cohort.
        epsilon: very small value to avoid division by zero.         
        
    """


    f = np.mean(np.abs(preds-score)/(uncs+epsilon))
    s = np.mean(uncs/(np.abs(preds-score)+epsilon))
    return s+f/s*f

def train(model, model_type, optimizer, train_loader):
    """
        Individual model training function. Since base learners in ensemble are trained individually, 
        each model is trained using this function

        Parameters
        ----------
        model: model to be trained.
        model_type: type of the model trained, options = {'RegGNN','PNA'}.
        optimizer: Optimizer to optimize training.
        train_loader: Data for training model.         
    """


    
    device = Config.EnsembleConstants.DEVICE
    num_epoch = Config.EnsembleConstants.EPOCH
    model = model.float().to(device)
    
    model.train()
    for epoch in range(num_epoch):
        epoch += 1
        print("\t\t\t\tEpoch Number: ", epoch)   
        for batch in train_loader:
            if(model_type=='RegGNN'):
                out = model(batch.x.to(device), to_dense(batch).adj.to(device))
            elif(model_type=='PNA'):    
                model.deg = compute_deg(train_loader)
                out = model(batch.x.to(device),
                            batch.edge_index.to(device),
                            batch.edge_attr.abs().to(device),
                            batch=1)

            loss = model.loss(out.view(-1, 1), batch.y.to(device).view(-1, 1))
            model.zero_grad()
            loss.backward()
            optimizer.step()

       
    return model

def test(model,model_type,test,device=torch.device('cpu')):
    """
        Individual model testing function. Since base learners in ensemble are trained individually, 
        each model is trained using this function

        Parameters
        ----------
        model: model which has already been trained to be tested.
        model_type: type of the model that will be tested.
                    options = {'RegGNN','PNA'}.
        test: data for testing model.          
    """


    device = Config.EnsembleConstants.DEVICE
    model.eval()
    with torch.no_grad():
        preds = []
        ground_truth = []
        for batch in test:
            if(model_type=='RegGNN'):
                out = model(batch.x.to(device), to_dense(batch).adj.to(device))

            elif(model_type=='PNA'):
                out = model(batch.x.to(device),
                                          batch.edge_index.to(device),
                                          batch.edge_attr.abs().to(device),
                                          batch=1)
            

            preds.append(out.cpu().data.numpy())
            loss = model.loss(out.view(-1, 1), batch.y.to(device).view(-1, 1))
            ground_truth.append(batch.y.cpu().long().numpy())

        preds = np.hstack(preds).reshape((len(test),))
        ground_truth = np.hstack(ground_truth).reshape((len(test),))
        test_mae = np.mean(np.abs(preds-ground_truth))
        
        
    return test_mae, preds, ground_truth


    
