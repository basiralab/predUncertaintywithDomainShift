import numpy as np
import torch
import pickle
from RegGNN.data_utils import load_dataset_pytorch,get_loaders,to_dense,to_sparse,simulate_dataset
from RegGNN.RegGNN import RegGNN
from PNA.PNA.model import PNANet,compute_deg
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from ensemble.ensemble_model import EnsembleModel
from ensemble.ensemble_uncertanity import *
from ensemble.ensemble_utils import e_u
from config import Config
import argparse

def constructEnsembleFromFile(filename,data):
    ensemble = []
    with open(filename) as f:
        for lines in f:
            variables = lines.split('\n')[0].split('/')
            if(variables[0]=='RegGNN'):
                ensemble.append({
                     'model':RegGNN(116, int(variables[1]), 1, int(variables[2]), int(variables[3]), float(variables[-1])),
                     'type': 'RegGNN'})
            else:
                ensemble.append({
                         'model':PNANet(116, int(variables[1]), 1, float(variables[-1]), aggrs=Config.PNA.AGGRS, scalers=Config.PNA.SCALERS,deg=compute_deg(data)).float().to(Config.EnsembleConstants.DEVICE),
                         'type': 'PNA'
                         })
        return ensemble

parser = argparse.ArgumentParser()

parser.add_argument('--data-source', default='simulated', choices=['simulate_data', 'predict'],
                    help="Simulates random data or predicts from already simulated data in path in config")
parser.add_argument('--mode', default='cv', choices=['cv', 'ds'],
                    help="Prediction process will get started by choices: \ncv: Cross Validation, \n ds: Domain Shift")
args = parser.parse_args()

#%%
if(args.data_source=='simulate_data'):
    
    conn, score = simulate_dataset(conn_mean=Config.CONNECTOME_MEAN, conn_std=Config.CONNECTOME_STD,
                                                       score_mean=Config.SCORE_MEAN, score_std=Config.SCORE_STD,
                                                       n_subjects=Config.N_SUBJECTS, seed=Config.DATA_SEED)

    torch.save(conn, f"{Config.DATA_FOLDER}{Config.CONNECTOME}")
    torch.save(score, f"{Config.DATA_FOLDER}{Config.SCORE}")
    print(f"Data is successfully generated to {Config.DATA_FOLDER}")

elif args.data_source == 'predict':
    data = load_dataset_pytorch(Config.CONNECTOME,Config.SCORE)
    print(f"Simulated data is successfully read from {Config.DATA_FOLDER}")
    ens = constructEnsembleFromFile("ensemble.txt",data)
    
    
    if(args.mode=='cv'):
        title = "Cross Validation Results of Ensemble"
        results = Cross_Validation_Ensemble(ens, data, True, seeds = Config.EnsembleConstants.SEED_LIST)
    elif args.mode == 'ds':
        title = "Domain Shift Results of Ensemble"
        results = Domain_Shift_Ensemble(ens, data, Config.EnsembleConstants.ALGO_LIST)
    else:
        raise Exception("Unknown parameter for mode.")
        
    err = e_u(results['mean_among_res'],results['unc_among_res'],results['score'])
    mae = mean_absolute_error(results['mean_among_res'],results['score'])
    
    
    ## Visualize output
    fig, ax0 = plt.subplots(1, 1)
    t = ax0.bar(["Proposed Error Metric","MAE"],[err,mae],color=("lightblue","purple"))
    ax0.bar_label(t)
    ax0.set_ylabel('Error Values')
    ax0.set_xlabel('Error Types')
    ax0.set_title(title)
    plt.show()
    
    print(f"Training process has succesfully finished. Results will be saved at {Config.RESULTS_FOLDER}")
    
    
    with open(f"{Config.RESULTS_FOLDER}preds.pkl", 'wb') as f:
        pickle.dump(results['mean_among_res'], f)
    with open(f"{Config.RESULTS_FOLDER}uncertanities.pkl", 'wb') as f:
        pickle.dump(results['unc_among_res'], f)

    print(f"Predictions for iq score and uncertanity are successfully saved at {Config.RESULTS_FOLDER}.")    
        
else:
    raise Exception("Unknown argument for data-source.")


#%%



