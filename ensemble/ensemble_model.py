import numpy as np
import torch

from config import Config
import ensemble.ensemble_utils  as ensemble_utils


class EnsembleModel():
    
    
    def __init__(self, ensemble,train,test):

        """
            Ensemble is created by using this class. Class has capability to train, test an ensemble easily.

            Parameters
            ----------
            ensemble: ensemble dictionary of models with the following format.
                     dictionary:{
                            model:Torch.nn
                            model_type:String|'RegGNN' or 'PNA'
                     }


            train: train loader to train ensemble model.
            test: test loader to test ensemble model.          
        """

        self.ensemble = ensemble
        self.train = train
        self.test = test
        
        
        self.models_test_mae = []
        self.models_test_preds = []
        self.models_test_scores = []

    
    def train_test_ensemble(self):
        """
            Train and test the ensemble independently. While training, Adam optimizer is used with Learning rate and Weight Decay
            in Config File. Feel Free to change to tune your model.
        """



        for i,model_dict in enumerate(self.ensemble):
          i=i+1
          print(f"\t\t\tModel {i} in Ensemble: ")
          model = model_dict['model']
          model_type = model_dict['type']
          
          
          optimizer = torch.optim.Adam(model.parameters(), 
                                       lr=Config.EnsembleConstants.LEARNING_RATE, 
                                       weight_decay=Config.EnsembleConstants.WEIGHT_DECAY)

          
          model_trained = ensemble_utils.train(model,  model_type,  optimizer,  self.train)
          test_mae, test_preds, test_scores = ensemble_utils.test(model_trained,  model_type, self.test)
          
          
          self.models_test_mae.append(test_mae)
          self.models_test_preds.append(test_preds)
          self.models_test_scores.append(test_scores)
        
        
        self.ensemble_test_preds,self.ensemble_test_mae = self.avg_ensemble(self,self.models_test_preds,self.models_test_scores[0])
        self.ensemble_test_uncs = np.std(self.models_test_preds,axis=0)

        
        
    @staticmethod
    def avg_ensemble(self,preds,score):
        """
            Takes the average outputted from each individual model for each subject in cohort.
            Returns prediction array and mean absolute error.
        """

        ensemble_result = np.mean(preds,axis=0)
        epoch_mae = np.mean(np.abs(ensemble_result-score))
        return ensemble_result,epoch_mae
    
    
    
      
    