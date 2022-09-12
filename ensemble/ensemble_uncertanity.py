# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:13:50 2022

@author: selim
"""

import numpy as np


from sklearn.cluster import *
from sklearn.model_selection import KFold
from sklearn.cluster  import MeanShift,AgglomerativeClustering,AffinityPropagation,KMeans,Birch,SpectralClustering

from config import Config
from ensemble.ensemble_model import EnsembleModel
import ensemble.ensemble_utils  as ensemble_utils

import copy





from RegGNN.data_utils import load_dataset_pytorch,get_loaders,to_dense,to_sparse
from RegGNN.RegGNN import RegGNN
from PNA.PNA.model import PNANet,compute_deg




def Cross_Validation_Ensemble(ENS_DICT, data, shuffle, seeds):
    """
            Training with Cross Validation(no domain shift) with the number of fold in Config file. 
            Training occurs after shuffling with all seed in seeds vector in order to diversify.

            EnsembleModel is initiliazed with the parameters.

            Returns a dictionary that contains the results:
                      'results':list of Ensemble Models for each seed,
                        'mean':list of predicted target output score for each seed,
                        'uncs':list of predicted uncertanity score for each seed,
                        'mean_among_res':mean among 'mean',
                        'unc_among_res':mean among 'uncs',
                        'score':Ground Truth for each subject.  

            Parameters
            ----------
            ENS_DICT: ensemble dictionary of models with the following format.
                     dictionary:{
                            model:Torch.nn
                            model_type:String|'RegGNN' or 'PNA'
                     }
            data: Data that will be folded and shuffled by seed.
            shuffle: Boolean variable to shuffle or not. Pass true to make shuffling.
            seeds: list of seed          
    """
    resultsCVs = []
    means = []
    uncs = []

    print("Training and Testing stage are starting...")
    for seed in seeds:   
        print(f"\tSeed: {seed}:")
        fold_id = 1
        mean = np.zeros((len(data),))
        unc = np.zeros((len(data),))
        score = np.zeros((len(data),))
        for train_idx, test_idx in KFold(Config.EnsembleConstants.CV_FOLD, shuffle=shuffle, random_state=seed).split(data):
            print(f"\t\tFold: {fold_id}...")
            train = [data[i] for i in train_idx]
            test = [data[i] for i in test_idx]
            
            ensemble = EnsembleModel(copy.deepcopy(ENS_DICT), train, test)
            ensemble.train_test_ensemble()
            resultsCVs.append(ensemble)
            
            mean[test_idx] = np.array(ensemble.ensemble_test_preds)
            unc[test_idx] =  np.array(ensemble.ensemble_test_uncs)
            score[test_idx] = np.array(ensemble.models_test_scores[0])
            
            fold_id +=1
        means.append(mean)
        uncs.append(unc)
        
    mean_among_res = np.mean(np.array(means),axis=0)
    unc_among_res = np.mean(np.array(uncs),axis=0)
    
    resultDict = {
            'results':resultsCVs,
            'mean':means,
            'uncs':uncs,
            'mean_among_res':mean_among_res,
            'unc_among_res':unc_among_res,
            'score':score
        }
    
    return resultDict

def domain_shift(data,method='KMeans'):
    y_values = []
    for item in data: 
        y_values.append(item.y.item())
    y_values = np.array(y_values)
    if(method=='MeanShift'):
        cl = MeanShift(bandwidth=7).fit(y_values.reshape(-1, 1))
    elif(method=='AgglomerativeClustering'):
        cl = AgglomerativeClustering(n_clusters=3).fit(y_values.reshape(-1, 1))
    elif(method=='AffinityPropagation'):
        cl = AffinityPropagation(damping=0.9382,random_state=1).fit(y_values.reshape(-1, 1))
    elif(method=='KMeans'):
        cl = KMeans(n_clusters=3, random_state=0).fit(y_values.reshape(-1, 1))
    elif(method=='BIRCH'):
        cl = Birch(threshold=0.01, n_clusters=3).fit(y_values.reshape(-1, 1))
    elif(method=='SpectralClustring'):
        cl = SpectralClustering(n_clusters=3).fit(y_values.reshape(-1, 1))
        
        
        
    cl.labels_[np.where(cl.labels_>2)]= 2
    class1 = (y_values[np.where(np.array(cl.labels_)==0)])
    class1_idx = np.where(np.array(cl.labels_)==0)[0]
    class2 = (y_values[np.where(np.array(cl.labels_)==1)])
    class2_idx = np.where(np.array(cl.labels_)==1)[0]
    class3 = (y_values[np.where(np.array(cl.labels_)==2)])
    class3_idx = np.where(np.array(cl.labels_)==2)[0]
    
    
    test1 = np.concatenate((class2_idx,class3_idx)),class1_idx#50
    test2 = np.concatenate((class1_idx,class3_idx)),class2_idx#100
    test3 = np.concatenate((class1_idx,class2_idx)),class3_idx#52
    tests = [test1,test2,test3]
    return tests

def Domain_Shift_Ensemble(ENS_DICT,data,cluster_algorithms = ['KMeans']):
    """
            Training with prepared simulation with Domain Shift with 3 folded test cases. 
            Training occurs after clustering the data with all algorithm in cluster_algorithms vector in order to diversify.

            EnsembleModel is initiliazed with the parameters.

            Returns a dictionary that contains the results:
                      'results':list of Ensemble Models for each seed,
                        'mean':list of predicted target output score for each seed,
                        'uncs':list of predicted uncertanity score for each seed,
                        'mean_among_res':mean among 'mean',
                        'unc_among_res':mean among 'uncs',
                        'score':Ground Truth for each subject.  

            Parameters
            ----------
            ENS_DICT: ensemble dictionary of models with the following format.
                     dictionary:{
                            model:Torch.nn
                            model_type:String|'RegGNN' or 'PNA'
                     }
            data: Data that will be folded and clustered by each cluster algorithm.
            cluster_algorithms: list of algorithms
                                Algorithms that are supported in this paper:    'AffinityPropagation',
                                                                                'KMeans',
                                                                                'BIRCH',
                                                                                'AgglomerativeClustering',
                                                                                'SpectralClustring'
                                                  
    """
    
        
    resultsDS = []
    train_indexes = []
    test_indexes = []
    means = []
    uncs = []
    print("Training and Testing stage are starting...")

    for algo in cluster_algorithms:  
 
        print(f"Clustering Algorithm: {algo}:")
        
        tests = domain_shift(data,method=algo)
        test_number = 1
        mean = np.zeros((len(data),))   
        unc = np.zeros((len(data),))
        score = np.zeros((len(data),))
        for train_idx, test_idx in tests:
            print(f"\t\tFold: {test_number}...")
            train = [data[i] for i in train_idx]
            test = [data[i] for i in test_idx]
            
            ensemble = EnsembleModel(copy.deepcopy(ENS_DICT), train, test)
            ensemble.train_test_ensemble()
            resultsDS.append(ensemble)
            train_indexes.append(train_idx)
            test_indexes.append(test_idx)
            
            mean[test_idx] = np.array(ensemble.ensemble_test_preds)
            unc[test_idx] =  np.array(ensemble.ensemble_test_uncs)
            score[test_idx] = np.array(ensemble.models_test_scores[0])
            
            test_number +=1
            
        means.append(mean)
        uncs.append(unc)
        
    mean_among_res = np.mean(np.array(means),axis=0)
    unc_among_res = np.mean(np.array(uncs),axis=0)
    
    resultDict = {
            'results':resultsDS,
            'means':means,
            'uncs':uncs,
            'mean_among_res':mean_among_res,
            'unc_among_res':unc_among_res,
            'score':score
        }
    
    return resultDict

