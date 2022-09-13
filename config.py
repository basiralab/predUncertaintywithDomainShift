from torch import device


class Config:
    # SIMULATED DATA OPTIONS
    DATA_FOLDER = './data/' #Data folder that simulation data will be read from and saved to.
    RESULTS_FOLDER = './results_folder/' #Data folder that prediction arrays will be saved to.
    CONNECTOME_MEAN = 0.0  # mean of the distribution from which connectomes will be sampled
    CONNECTOME_STD = 1.0  # std of the distribution from which connectomes will be sampled
    SCORE_MEAN = 90.0  # mean of the distribution from which scores will be sampled
    SCORE_STD = 10.0  # std of the distribution from which scores will be sampled
    N_SUBJECTS = 202  # number of subjects in the simulated data
    ROI = 116  # number of regions of interest in brain graph
    CONNECTOME="connectome.ts" # tensor filename of the simulated connectome.
    SCORE = "score.ts" # tensor filename of the simulated score.
    
    DATA_SEED = 1  # random seed for data creation
    MODEL_SEED = 1  # random seed for models
    

    # PNA OPTIONS
    class PNA:
        SCALERS = ["identity", "amplification", "attenuation"]  # scalers used by PNA
        AGGRS = ['sum', 'mean', 'var', 'max']  # aggregators used by PNA


    #ENSEMBLE OPTIONS
    class EnsembleConstants():
        
        SEED_LIST = [5]#,6,7,8,9]  seeds used during shuffling.
        ALGO_LIST = ['AffinityPropagation','KMeans','BIRCH','AgglomerativeClustering','SpectralClustring']# algorithms used during clustering.
        CV_FOLD = 3 # Number of Fold in Cross Validation  
        EPOCH = 1 #Number of Epoch
        DEVICE = device('cpu') #Device which training and testing will be on.
        LEARNING_RATE = 1e-3 # Learning rate for optimizer
        WEIGHT_DECAY = 5e-4 # Weigt Decay for optimizer
    
    
    