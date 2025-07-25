## A bunch of functions required to run the reproducible Jupyter Notebook 
## for the paper's plot outputs and also to run some of the function in
## docs/source/content/examples

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from modAL.models import ActiveLearner
from modAL.uncertainty import * 
import tensorflow as tf
from .glad import custom_binary_crossentropy_loss
from .loda_utils import LODA_OAT

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from skorch import NeuralNetClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from scipy.stats import zscore


###################################################################################################
###################################################################################################

# class SimpleNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim=32):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 2)  # Binary classification

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)  # No softmax; skorch handles that

# class SimpleNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim=None):
#         super(SimpleNN, self).__init__()
#         if hidden_dim is None:
#             hidden_dim = max(50, 3 * input_dim)

#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),  # or nn.LeakyReLU()
#             nn.Dropout(0.3),  # optional but helps
#             nn.Linear(hidden_dim, 2)
#         )

#     def forward(self, x):
#         return self.net(x)

# class SimpleNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim=None):
#         super(SimpleNN, self).__init__()
#         if hidden_dim is None:
#             hidden_dim = max(50, 3 * input_dim) 

#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, 2)
#         )

#         # Custom weight initialization
#         for m in self.net:
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         return self.net(x)

# class SimpleNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim=None):
#         super(SimpleNN, self).__init__()
#         if hidden_dim is None:
#             hidden_dim = max(50, 3 * input_dim)
#         hidden_dim2 = max(30, hidden_dim // 2)  # second hidden layer smaller

#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LeakyReLU(0.1),
#             #nn.Dropout(0.1),

#             nn.Linear(hidden_dim, hidden_dim2),
#             nn.LeakyReLU(0.1),
#             #nn.Dropout(0.1),

#             nn.Linear(hidden_dim2, 2)
#         )

#         # Custom weight initialization
#         for m in self.net:
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         return self.net(x)

###################################################################################################
###################################################################################################

def InitActiveAGG(X_old = None,Y_old = None,n_data_min = 100, models=None):

    ###############################################################################################
    # Description: This function runs before the main "ActiveAGG" function. It treats the existence 
    #              or absense of data provided from the past.
    #
    # Possible arguments in **kwargs: 
    #    X_old : a numpy array of shape (number of old data) x (number of data dimensions) 
    #    Y_old : either not in the parameters (no old data) or a list of items
    #            with values 0, 1, or nan. Note that if
    #            X_old exists, we require Y_old to exist, even if it is filled completely
    #            with nan (no labeled data)
    #    n_data_min : minimum number of data points required before calculating any anomaly scores
    #                 (if not provided, we set this to the default = 100)
    #    models : a dictionary of (unsupervised) models + parameters from Tadkit that should be 
    #             used to score data
    #
    # Outputs :
    #     X_lab 
    #     Y_lab
    #     all_labeled_scores
    ###############################################################################################
    
    if X_old is None:
        #X_old = pd.DataFrame()
        X_old = np.empty((0,0))
    if Y_old is None:
        Y_old = []
    if models is None:
        models = dict()
    
    #If there is no old data:
    if X_old.size == 0:
        
        X_lab = np.empty([0,0])
        Y_lab = []  
        all_labeled_scores = np.empty([0,0])
        Y_old = []
        
        
    #Else there is old data: 
    else:
        #if np.shape(Y_old) == (0,0):
        if len(Y_old) == 0:
            raise ValueError('Y_old must be provided since X_old was provided.')
        
        #deal with all cases: (no labels, some labels, fully labeled):
        
        #which_lab = [i for i in range(len(Y_old)) if Y_old[i] is not np.nan]
        which_lab = [i for i in range(len(Y_old)) if not np.isnan(Y_old[i])]
        
        #If none of the old data is labeled:
        if len(which_lab) == 0:
            X_lab = np.empty([0,0])
            Y_lab = [] 
            all_labeled_scores = np.empty([0,len(models.items())])
            
        #Else some of it is labeled, and so:
        else: 
            X_lab = X_old[which_lab,:]
            #X_lab = X_old.iloc[which_lab]
            #X_lab = X_lab.to_numpy()
            Y_lab = [Y_old[j] for j in which_lab]
            
            
            # If there is at least one item with label 0 OR one item with label 1,
            # we can start to calculate scores for each available method: 
            if sum([1 for i in range(len(Y_lab)) if Y_lab[i]==0]) > 0 or sum([1 for i in range(len(Y_lab)) if Y_lab[i]==1]) > 0:
                                
                # How many labeled data are there?:
                #n_lab = sum([1 for i in range(len(Y_lab)) if Y_lab[i]==0]) + sum([1 for i in range(len(Y_lab)) if Y_lab[i]==1])
                
                # Since there are labeled data from each class, now we want to check that we have 
                # enough data OVERALL for doing meaningful anomaly detection already. If we do not
                # then no scores are calculated by this initialization function
                
                if np.shape(X_old)[0] >= n_data_min:
                    
                    # Define arrays to stock all scores:
                    all_scores = np.empty([np.shape(X_old)[0],len(models.items())])

                    # We run through the unsupervised methods, one by one, to output scores:
                    
                    if len(models.items()) == 0:
                        raise ValueError('We require a non-empty dictionary called "models" in order to run unsupervised anomaly detection models.')      
                    
                    # for i, (name, model) in enumerate(models.items()):
                    #     model.fit(X_old)
                    #     print('X old size:',np.shape(X_old))
                    #     print('X_old:',X_old)
                    #     y_score = model.score_samples(X_old)
                    #     y_score.dtype = np.float64
                    #     print('Y scores:',y_score)
                    #     print('Y score shape:',np.shape(y_score))
                    #     all_scores[:,i] = y_score.squeeze()

                    models["LocalOutlierFactor"] = LocalOutlierFactor(novelty=True)

                    for i, (name, model) in enumerate(models.items()):
                        print(f"\nModel: {name}")
                        print("X_old shape before fit:", X_old.shape)

                        print(f"Model: {name} | Is LOF: {isinstance(model, LocalOutlierFactor)} | novelty: {getattr(model, 'novelty', 'N/A')}")
                        model.fit(X_old)
                        y_score = model.score_samples(X_old)
                    
                        print("y_score shape:", y_score.shape)
                        print('y_score values:',y_score)
                    
                        if y_score.shape[0] != all_scores.shape[0]:
                            raise ValueError(f"Model {name} returned wrong number of scores: {y_score.shape[0]} instead of {all_scores.shape[0]}")
                    
                        #y_score.dtype = np.float64
                        y_score = y_score.astype(np.float64)
                        print("y_score shape:", y_score.shape)
                        print('y_score values:',y_score)
                        yscoresqueeze = y_score.squeeze()
                        print("y_score squeeze shape:", yscoresqueeze.shape)
                        print('y_score squeeze values:',yscoresqueeze)
                        print('Current shape of all_scores:',np.shape(all_scores))
                        all_scores[:, i] = y_score.squeeze()

                    
                    # We then extract the subset of the array with the scores for the labeled data:
                    all_labeled_scores = np.take(all_scores, which_lab, 0)
                    
                
                else:
                    all_labeled_scores = np.empty([0,0])
                    
            
            else:    
                all_labeled_scores = np.empty([0,0])
                
    
    return X_lab, Y_lab, all_labeled_scores

###################################################################################################
###################################################################################################

def InitActiveAGG_2(X_old = None,Y_old = None,n_data_min = 100, models=None,curr_method = None,**kwargs):

    ###############################################################################################
    # Description: This function runs before the main "ActiveAGG" function. It treats the existence 
    #              or absense of data provided from the past.
    #
    # Possible arguments in **kwargs: 
    #    X_old : a numpy array of shape (number of old data) x (number of data dimensions) 
    #    Y_old : either not in the parameters (no old data) or a list of items
    #            with values 0, 1, or nan. Note that if
    #            X_old exists, we require Y_old to exist, even if it is filled completely
    #            with nan (no labeled data)
    #    n_data_min : minimum number of data points required before calculating any anomaly scores
    #                 (if not provided, we set this to the default = 100)
    #    models : a dictionary of (unsupervised) models + parameters from Tadkit that should be 
    #             used to score data
    #    supervised_method : a choice of one method that can do supervised 
    #                        learning on score vectors which have known labels 
    #                        (0 non-anomaly, 1 anomaly)
    #
    # Outputs :
    #     : lsddlsf
    ###############################################################################################
    
    now_scores = kwargs.get('now_scores', None)
    
    if X_old is None:
        #X_old = pd.DataFrame()
        X_old = np.empty((0,0))
    if Y_old is None:
        Y_old = []
    if models is None:
        models = dict()
    
    #If there is no old data:
    if X_old.size == 0:
        
        X_lab = np.empty([0,0])
        Y_lab = []  
        all_labeled_scores = np.empty([0,0])
        Y_old = []
        
        
    #Else there is old data: 
    else:
        #if np.shape(Y_old) == (0,0):
        if len(Y_old) == 0:
            raise ValueError('Y_old must be provided since X_old was provided.')
        
        #deal with all cases: (no labels, some labels, fully labeled):
        
        #which_lab = [i for i in range(len(Y_old)) if Y_old[i] is not np.nan]
        which_lab = [i for i in range(len(Y_old)) if not np.isnan(Y_old[i])]
        
        #If none of the old data is labeled:
        if len(which_lab) == 0:
            X_lab = np.empty([0,0])
            Y_lab = [] 
            all_labeled_scores = np.empty([0,len(models.items())])
            
        #Else some of it is labeled, and so:
        else: 
            X_lab = X_old[which_lab,:]
            #X_lab = X_old.iloc[which_lab]
            #X_lab = X_lab.to_numpy()
            Y_lab = [Y_old[j] for j in which_lab]
            
            
            # If there is at least one item with label 0 OR one item with label 1,
            # we can start to calculate scores for each available method: 
            if sum([1 for i in range(len(Y_lab)) if Y_lab[i]==0]) > 0 or sum([1 for i in range(len(Y_lab)) if Y_lab[i]==1]) > 0:
                                
                # How many labeled data are there?:
                #n_lab = sum([1 for i in range(len(Y_lab)) if Y_lab[i]==0]) + sum([1 for i in range(len(Y_lab)) if Y_lab[i]==1])
                
                # Since there are labeled data from each class, now we want to check that we have 
                # enough data OVERALL for doing meaningful anomaly detection already. If we do not
                # then no scores are calculated by this initialization function
                
                if np.shape(X_old)[0] >= n_data_min:
                    
                    # Define arrays to stock all scores:
                    all_scores = np.empty([np.shape(X_old)[0],len(models.items())])

                    # We run through the unsupervised methods, one by one, to output scores:
                    
                    if len(models.items()) == 0:
                        raise ValueError('We require a non-empty dictionary called "models" in order to run unsupervised anomaly detection models.')
                    
                    
                    ####### KEEP IN CASE NEED IT #############################################
                    ####### OLD VERSION TO BE FIXED !!!!!!! ##################################
                    #formalizer = PandasFormalizer(data_df=X_old, dataframe_type="synchronous")
                    #results = {}
                    #base_query = formalizer.default_query()
                    #X = formalizer.formalize(**base_query)
                    #base_query["target_period"] = (data.index[0])
                    ##########################################################################
                    
                    ##########################################################################

                    if curr_method=="RandomScore":
                        all_labeled_scores = np.take(now_scores, which_lab, 0)

                    else:
                        for i, (name, model) in enumerate(models.items()):
                            model.fit(X_old)
                            y_score = model.score_samples(X_old)
                            y_score.dtype = np.float64
                            all_scores[:,i] = y_score.squeeze()
    
                        #print('all scores shape:',np.shape(all_scores))
                        #print('now_scores shape:',np.shape(now_scores))
                        # We then extract the subset of the array with the scores for the labeled data:
                        
                        
                        all_labeled_scores = np.take(all_scores, which_lab, 0)
                    
                
                else:
                    all_labeled_scores = np.empty([0,0])
                    
            
            else:    
                all_labeled_scores = np.empty([0,0])
                
    
    return X_lab, Y_lab, all_labeled_scores  
    

###################################################################################################
###################################################################################################


def ActiveAGG(X_new = None, X_old = None, X_lab = None, Y_lab = None, all_labeled_scores = None, models=None,supervised_method = None,n_data_min = 100,n_data_max = 2000,min_n_labeled = 5,n_send=2,pc_top = 0.5,min_n_nom=5,min_n_anom=1,tau_exp=0.01):

    ###############################################################################################
    # Description: This function runs active anomaly detection on time-series batch data. Each
    # time it is called, it "eats" some new data, it runs all the methods to get scores for each
    # new data time point, it uses active learning to propose some of these to an expert, and it
    # predicts new anomalies using the current version of the supervised machine learning model. 
    #
    # Arguments: 
    #    X_new : new numpy array batch of B time-series data of size B x number of data dimensions
    #    X_old : numpy array of all (or most recent) past data
    #    X_lab : numpy array of the current labeled data 
    #    Y_lab : list of the current labels of the labeled data
    #    all_labeled_scores : current numpy array of all score vectors that have been labeled so far.
    #    models : a dictionary of (unsupervised) models + parameters from Tadkit that should be 
    #             used to score data
    #    supervised_method : a choice of one method that can do supervised 
    #                        learning on score vectors which have known labels 
    #                        (0 non-anomaly, 1 anomaly)
    #    n_data_min : minimum number of data points required before calculating any anomaly scores
    #                 (if not provided, we set this to the default = 100)
    #    n_data_max : maximum number of most recent time points to send to each method (in order to
    #                 keep run times faster). 
    #    min_n_labeled : minimum number of labeled data required before running supervised learning.
    #    n_send : total number of unlabeled items that can be sent to the expert for any reason 
    #             during any loop.
    #    pc_top : the percentage of the n_send items that are used to predict anomalies (as a fraction, 
    #             e.g. 0.6)
    #    min_n_nom : minimum number of labeled nominals required before doing supervised and
    #                active learning
    #    min_n_anom : minimum number of labeled anomalies required before doing supervised and
    #                active learning
    #    tau_exp : expected fraction of anomalies (in reality we don't know it, just that we expect it
    #              to be small)
    #
    #
    # Remarks : 
    #    1) One issue that may come up is that over time, we may have too much labeled data to 
    #       efficiently run our maching learning model, even if we started in the semi-supervised
    #       setting.
    #    2) The list "methods" needs to be IDENTICAL to the list "methods" in InitActiveAGG
    ###############################################################################################
    
    if X_new is None:
        #X_new = pd.DataFrame()
        X_new = np.empty((0,0))
        
    if X_old is None:
        #X_old = pd.DataFrame()
        X_old = np.empty((0,0))
        
    if models is None:
        models = dict()
        
    if X_lab is None:
        X_lab = np.empty([0,0])
        
    if Y_lab is None:
        Y_lab = []
        
    if all_labeled_scores is None:
        all_labeled_scores = np.empty([0,0])
    
    # Errors and warnings:
    
    # 1) X_new needs to be non-empty:
    if X_new.size == 0:
        raise ValueError('New data "X_new" cannot be empty.')
            
    # 2) If there is old data, check that the 2nd dimension of its array is the same as that
    #    of the new data array:
    if X_old.size != 0:
        if np.shape(X_old)[1] != np.shape(X_new)[1]:
            raise ValueError('Data dimension of X_old and X_new need to be the same.')
            
    # 3) We need to have at least one unsupervised model that can output scores:
    if len(models.items()) == 0:
        raise ValueError('We require a non-empty dictionary called "models" in order calculate anomaly scores.')
        
    # 4) n_data_min must be no bigger than n_data_max:
    if n_data_min > n_data_max:
        raise ValueError('"n_data_min" cannot be greater than "n_data_max".')
        
    # 5) The new data X_new must be shorter than "n_data_max" since otherwise we cannot
    #    predict the label of all of the new data.
    if np.shape(X_new)[0] > n_data_max:
        raise ValueError('The number of new data needs to be less than "n_data_max".')
        
    # 6) Need to have one supervised method, and one only, provided.
    if supervised_method is None:
        raise ValueError('One supervised_method must be supplied to the function.')
        
    
    #Initialize:
    supervised_indices = []
    indices_to_expert = []
    #n_send = -1
    
    #Next new batch size:
    myB = np.shape(X_new)[0]
    
    #Calculate n_top and n_active, the number of items in each loop that will be sent to
    #the expert for anomaly prediction and active learning, respectively.
    n_top = int(np.ceil(pc_top*n_send))
    n_active = n_send - n_top
    
    #Concatenate the old data to the new data:
    X_curr = np.concatenate([X_old,X_new],axis=0)
    
    # Send at most the last n_data_max rows to the methods for scoring (in order to keep
    # run time reasonable):
    if np.shape(X_curr)[0] > n_data_max:
        X_curr = X_curr[-n_data_max:,:]

    
    #######################################################################################
    # CALCULATING SCORES ##################################################################
    #######################################################################################
    
    # We require there to be at least n_data_min data points before calculating scores
    if np.shape(X_curr)[0] >= n_data_min:

                    
        # Define array to stock all current scores. Note that if X_curr contains data from the
        # original X_old, scores will be calculated ANEW here BUT if there had been sufficient 
        # labeled data in X_old originally to start to fill in "all_labeled_scores" previously,
        # WE DO NOT UPDATE "all_labeled_scores" based on the recalculation of scores here.
        
        # Calculate scores for all current data and all models: 
        all_scores = np.empty([np.shape(X_curr)[0],len(models.items())])

        # We run through the unsupervised methods, one by one, to output scores:
        
        
        ################# KEEP JUST IN CASE !!!! ###################################
        #formalizer = PandasFormalizer(data_df=X_curr, dataframe_type="synchronous")
        #results = {}
        #base_query = formalizer.default_query()
        #X = formalizer.formalize(**base_query)
        #base_query["target_period"] = (data.index[0])
        ############################################################################
        ############################################################################
        
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_curr)
            y_score = model.score_samples(X_curr)
            y_score.dtype = np.float64
            all_scores[:,i] = y_score.squeeze()
        
        # For simplicity, we only keep the scores corresponding to new data:
        all_scores = all_scores[-myB:,:]
        #print('all scores top:',all_scores[:10,])
        #print('Length of all_scores is: ',np.shape(all_scores)[0])
        
        # Supervised learning interlude: if we have sufficient labeled data 
        # (either from the initialization step or from PREVIOUS loops of the
        # present function), AND we have at least five (for example) labeled item from each class, 
        # we run the machine learning method chosen. It is likely that the set
        # of allowed methods here must be able to use "predict" in Python.
        
        
        if len(Y_lab) >= min_n_labeled and sum([1 for i in range(len(Y_lab)) if Y_lab[i]==0]) >= min_n_nom and sum([1 for i in range(len(Y_lab)) if Y_lab[i]==1]) >= min_n_anom:
                
            # We find the method in a list and run it:
            
            #########################################################################################
            
            # 1) Random forest classifier:
            if supervised_method == 'RandomForestClassifier':
                #We want to use the classifier set-up first since we can weigh the classes differently.
                #But afterwards we really want to treat it as regression in order to propose candidates
                #to an expert.

                r0 = 1-tau_exp  # Proportion of class 0 expected
                weights = {0: r0, 1: 1 - r0}
                RFC = RandomForestClassifier(class_weight=weights)
                #RFC = RandomForestClassifier(class_weight='balanced')
                learned_model = RFC.fit(all_labeled_scores, Y_lab)
                # Predicted probabilities for the positive class
                new_preds = learned_model.predict_proba(all_scores)[:, 1]
                #print('new_preds',new_preds)
                
                #print('new_preds',new_preds)

            if supervised_method == 'LogisticRegression':
                #r0 = 1-tau_exp  # Proportion of class 0 expected
                #weights = {0: r0, 1: 1 - r0}          
                #LRC = LogisticRegression(class_weight=weights)
                #LRC = LogisticRegression(class_weight='balanced',penalty='l2')
                LRC = LogisticRegression(class_weight='balanced')
                
                learned_model = LRC.fit(all_labeled_scores, Y_lab)
                # Predicted probabilities for the positive class
                new_preds = learned_model.predict_proba(all_scores)[:, 1]

            # if supervised_method == 'NeuralNet':

            #     # Step 1: Convert Y_lab and compute class weights
            #     y_array = np.asarray(Y_lab, dtype=np.int64)
            #     class_counts = np.bincount(y_array)
            #     total = class_counts.sum()
            #     class_weights = total / (2.0 * class_counts)
            #     class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            
            #     # Step 2: Z-score normalize the labeled scores and all scores independently
            #     all_labeled_scores_norm = zscore(all_labeled_scores, axis=0)
            #     all_scores_norm = zscore(all_scores, axis=0)
            
            #     # Step 3: Define model with weighted loss
            #     net = NeuralNetClassifier(
            #         SimpleNN,
            #         module__input_dim=all_labeled_scores.shape[1],
            #         max_epochs=30,
            #         lr=0.01,
            #         verbose=0,
            #         callbacks=[],
            #         train_split=None,
            #         criterion=nn.CrossEntropyLoss,
            #         criterion__weight=class_weights_tensor
            #     )
            
            #     learned_model = net.fit(
            #         all_labeled_scores_norm.astype(np.float32),
            #         y_array
            #     )
            
            #     # Predict probabilities on normalized scores
            #     new_preds = learned_model.predict_proba(all_scores_norm.astype(np.float32))[:, 1]
                                            
                

            
            # We will then use this learned model to predict the label of all of the data in X_curr.
            # corresponding to new data only:
            #new_preds = learned_model.predict(all_scores)
            #print(new_preds)

            #n_new_pred_anom = sum([r==1 for r in new_preds])
            
            # How good is this learned model 
            
            # Given that these predictions are set up so that the higher the prediction, the more
            # we suspect a data point is an anomaly, it makes sense to concentrate on sending the 
            # "highest" predictions to an expert who is looking for anomalies. There are a variety
            # of ways this could be done. For now, we hard code one method, which is to send the
            # top pc_pred fraction of predictions to the expert (or at a minimum, 1 prediction)
            
            #n_send = int(max(np.floor(pc_pred*myB),1))
            
            #Or we send all predicted anomalies:
            #n_send = n_new_pred_anom
            
            # We have to then sort "new_preds" and get the indices of the n_send largest:
            
            sorted_indices = np.argsort(new_preds)
            #print('sorted indices',sorted_indices)
            if n_top > 0:
                supervised_indices = sorted_indices[-n_top:].tolist()
            else:
                supervised_indices = []

            #print('Supervised indices:',supervised_indices)
            
            #sorted_indices = sorted(range(len(new_preds)), key=lambda k: new_preds[k])
            #supervised_indices = sorted_indices[-n_top:] 
            #print('Supervised_indices:',supervised_indices)
            #Print the corresponding values of x in the 1d case:
            #X_end = X[-myB:,:]
            #print('The x values predicted to be anomalies: ',X_end[supervised_indices])
        
        else: 
            learned_model = None


        #######################################################################################
        # ACTIVE LEARNING #####################################################################
        #######################################################################################

        # Here we implement a module that performs active learning. There are an enormous 
        # number of ways to go about this, whether it be which strategy to use, but also from
        # which pool of data do we look for candidates.

        # We shall do active learning by wrapping around the modAL Python package, which
        # implements active learning by itself (mostly) wrapping around methods in ScikitLearn.

        # For the moment, we use only the data in
        # X_new as potential data to give to the expert, given the labeled data we already have
        # from previous loops (if any). In the next phase of code development we can perhaps
        # also include ALL previous unlabeled data in this section.

        # First we need to know if there is already some labeled data or not. If there is
        # so far NO labeled data, or only data labeled from one class, we propose to use 
        # random sampling. If there is already labeled data from both classes, we shall
        # use the 

        active_indices = []
        curr_all_scores = all_scores

        if len(Y_lab) > min_n_labeled and sum([1 for i in range(len(Y_lab)) if Y_lab[i]==0]) >= min_n_nom and sum([1 for i in range(len(Y_lab)) if Y_lab[i]==1]) >= min_n_anom:
            
            
            curr_all_labeled_scores = all_labeled_scores
            curr_Y_lab = Y_lab
            
            # Set up the chosen active learning strategy (default in modAL is uncertainty sampling)
            if supervised_method == 'RandomForestClassifier':
                r0 = 1-tau_exp  # Proportion of class 0 expected
                weights = {0: r0, 1: 1 - r0}
                learner = ActiveLearner(
                #estimator=RandomForestClassifier(class_weight='balanced'),
                estimator=RandomForestClassifier(class_weight=weights),
                #query_strategy=uncertainty_sampling,
                query_strategy = margin_sampling,
                #query_strategy = entropy_sampling,
                X_training=curr_all_labeled_scores, y_training=curr_Y_lab
                )
                
            if supervised_method == 'LogisticRegression':
                #r0 = 1-tau_exp  # Proportion of class 0 expected
                #weights = {0: r0, 1: 1 - r0}
                learner = ActiveLearner(
                #estimator=LogisticRegression(class_weight=weights),
                #estimator=LogisticRegression(class_weight='balanced',penalty='l2'),
                estimator=LogisticRegression(class_weight='balanced'),
                #query_strategy=uncertainty_sampling,
                query_strategy = margin_sampling,
                #query_strategy = entropy_sampling,
                X_training=curr_all_labeled_scores, y_training=curr_Y_lab
                )

            # if supervised_method == 'NeuralNet':
    
            #     # Step 1: Convert labels to NumPy array
            #     y_array = np.asarray(curr_Y_lab, dtype=np.int64)
                
            #     # Step 2: Compute class weights
            #     class_counts = np.bincount(y_array)
            #     total = class_counts.sum()
            #     class_weights = total / (2.0 * class_counts)
            #     class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
                
            #     # Minimal change: normalize the training features here
            #     curr_all_labeled_scores_norm = zscore(curr_all_labeled_scores, axis=0)
                
            #     estimator = NeuralNetClassifier(
            #         SimpleNN,
            #         module__input_dim=curr_all_labeled_scores.shape[1],
            #         max_epochs=20,
            #         lr=0.01,
            #         verbose=0,
            #         callbacks=[],
            #         train_split=None,
            #         criterion=nn.CrossEntropyLoss,           
            #         criterion__weight=class_weights_tensor      
            #     )
                
            #     # Create ActiveLearner with normalized training data
            #     learner = ActiveLearner(
            #         estimator=estimator,
            #         query_strategy=margin_sampling,
            #         X_training=curr_all_labeled_scores_norm.astype(np.float32),
            #         y_training=y_array
            #     )
        
                
            #Since we want to do active learning using the modAL package, unfortunately it only 
            #outputs one candidate at a time, so there is a whole lot of work to do to make sure
            #nothing stupid happens with indices as we iteratively output one candidate at a time
            #until we get to n_active.

            true_indices = list(range(0,np.shape(curr_all_scores)[0]))

            if supervised_method == 'NeuralNet':
                curr_all_scores_norm = zscore(curr_all_scores, axis=0).astype(np.float32)

            i = 0
            while i < n_active:   
                          
                #query_idx, query_inst = learner.query(curr_all_scores)

                if supervised_method == 'NeuralNet':
                    query_idx, query_inst = learner.query(curr_all_scores_norm)
                else:
                    query_idx, query_inst = learner.query(curr_all_scores.astype(np.float32))

                
                #print('query_idx = ',query_idx)
                #print('query_inst = ',query_inst)
                query_idx = int(query_idx[0])

                # Get the actual index:
                actual_index = true_indices[query_idx]
                #print('Actual index:',actual_index)

                # See whether it was already in supervised_indices and if so, skip it.
                if actual_index in supervised_indices:
                    #print('It was a repeat')
                    #Remove this index from true_indices:
                    del true_indices[query_idx]
                    # Remove this guy from the data and loop back:
                    curr_all_scores = np.delete(curr_all_scores, query_idx, axis=0)
                    continue
                    
                #else:
                # Append new index to list:               
                active_indices.append(true_indices[query_idx])
                
                
                #Remove this index from true_indices:
                del true_indices[query_idx]
                
                # Remove this guy from the data and loop back:
                curr_all_scores = np.delete(curr_all_scores, query_idx, axis=0)
                #del curr_Y_lab[query_idx]

                i += 1
            #print('Active indices:',active_indices)


        else:
            print('The condition was not satisfied!')
            # Has not been updated to make sure we get the right total number of items to send to the expert, 
            # since in our simulations (which this version of the function is for), this doesn't happen.
            # There is no (or not enough) labeled data, so we do random sampling:
            my_random_permute = np.random.permutation(np.shape(curr_all_scores)[0])
            #learner = ActiveLearner(
            #estimator=RandomForestClassifier(class_weight='balanced'),    #irrelevant
            #query_strategy=random_sampling,
            #X_training=X_curr, y_training=np.ones((np.shape(X_curr)[0],1))     #irrelevant
            #)

            # query for labels
            #query_idx, query_inst = learner.query(curr_all_scores)
            # Make sure that the indices of the selected data are integers and turn this into a list:
            
            active_indices = [int(my_random_permute[i]) for i in range(n_active)]


        #print('active_indices:',type(active_indices))
        #print('supervised_indices:',type(supervised_indices))
        # So now we have both data from the supervised prediction, AND data from the active
        # learning, to give to the expert for labeling. We need to merge these two lists of 
        # indices as they may have overlaps
        
        
        
        indices_to_expert = supervised_indices + active_indices
        #print('superv_ind_merged_with_activ_ind:',indices_to_expert)
        indices_to_expert = sorted(set(indices_to_expert))
    
    
    
        #######################################################################################
        # UPDATING PARAMETERS #################################################################
        #######################################################################################
    
        
        
        
        # Update all_labeled_scores:
        if np.shape(all_labeled_scores) != (0,0): 
            all_labeled_scores = np.concatenate([all_labeled_scores,all_scores[indices_to_expert,:]],axis=0)


        #print('merged indices sent to expert:',indices_to_expert)

    
        # Update X_lab:
        if np.shape(X_lab) != (0,0):
            X_new_labeled = X_new[indices_to_expert,:]
            X_lab = np.concatenate([X_lab,X_new_labeled],axis=0)
        else:
            X_lab = X_new[indices_to_expert,:]
    
    # Update X_old:
    X_old = np.concatenate([X_old,X_new],axis=0)
    
    # Retain at most n_data_max points in X_old:
        
    if np.shape(X_old)[0] > n_data_max:
        X_old = X_old[-n_data_max:,:]
        
    
    
    return X_old, X_lab, all_labeled_scores, indices_to_expert, learned_model, supervised_indices 
    

##################################################################################################
##################################################################################################

def ActiveAGG_2(X_new = None, X_old = None, X_lab = None, Y_lab = None, all_labeled_scores = None, models=None,supervised_method = None,n_data_min = 100,n_data_max = 2000,min_n_labeled = 5,n_send=2,pc_top = 0.5,min_n_nom=5,min_n_anom=1,tau_exp=0.01,curr_method = None,**kwargs):

    ###############################################################################################
    # Description: This function runs active anomaly detection on time-series batch data. Each
    # time it is called, it "eats" some new data, it runs all the methods to get scores for each
    # new data time point, it uses active learning to propose some of these to an expert, and it
    # predicts new anomalies using the current version of the supervised machine learning model. 
    #
    # Arguments: 
    #    X_new : new numpy array batch of B time-series data of size B x number of data dimensions
    #    X_old : numpy array of all (or most recent) past data
    #    X_lab : numpy array of the current labeled data 
    #    Y_lab : list of the current labels of the labeled data
    #    all_labeled_scores : current numpy array of all score vectors that have been labeled so far.
    #    models : a dictionary of (unsupervised) models + parameters from Tadkit that should be 
    #             used to score data
    #    supervised_method : a choice of one method that can do supervised 
    #                        learning on score vectors which have known labels 
    #                        (0 non-anomaly, 1 anomaly)
    #    n_data_min : minimum number of data points required before calculating any anomaly scores
    #                 (if not provided, we set this to the default = 100)
    #    n_data_max : maximum number of most recent time points to send to each method (in order to
    #                 keep run times faster). 
    #    min_n_labeled : minimum number of labeled data required before running supervised learning.
    #    n_send : total number of unlabeled items that can be sent to the expert for any reason 
    #             during any loop.
    #    pc_top : the percentage of the n_send items that are used to predict anomalies (as a fraction, 
    #             e.g. 0.6)
    #    min_n_nom : minimum number of labeled nominals required before doing supervised and
    #                active learning
    #    min_n_anom : minimum number of labeled anomalies required before doing supervised and
    #                active learning
    #    tau_exp : expected fraction of anomalies (in reality we don't know it, just that we expect it
    #              to be small)
    #
    #
    # Remarks : 
    #    1) One issue that may come up is that over time, we may have too much labeled data to 
    #       efficiently run our maching learning model, even if we started in the semi-supervised
    #       setting.
    #    2) The list "methods" needs to be IDENTICAL to the list "methods" in InitActiveAGG
    ###############################################################################################
    
    if X_new is None:
        #X_new = pd.DataFrame()
        X_new = np.empty((0,0))
        
    if X_old is None:
        #X_old = pd.DataFrame()
        X_old = np.empty((0,0))
        
    if models is None:
        models = dict()
        
    if X_lab is None:
        X_lab = np.empty([0,0])
        
    if Y_lab is None:
        Y_lab = []
        
    if all_labeled_scores is None:
        all_labeled_scores = np.empty([0,0])
    
    # Errors and warnings:
    
    # 1) X_new needs to be non-empty:
    if X_new.size == 0:
        raise ValueError('New data "X_new" cannot be empty.')
            
    # 2) If there is old data, check that the 2nd dimension of its array is the same as that
    #    of the new data array:
    if X_old.size != 0:
        if np.shape(X_old)[1] != np.shape(X_new)[1]:
            raise ValueError('Data dimension of X_old and X_new need to be the same.')
            
    # 3) We need to have at least one unsupervised model that can output scores:
    if len(models.items()) == 0:
        raise ValueError('We require a non-empty dictionary called "models" in order calculate anomaly scores.')
        
    # 4) n_data_min must be no bigger than n_data_max:
    if n_data_min > n_data_max:
        raise ValueError('"n_data_min" cannot be greater than "n_data_max".')
        
    # 5) The new data X_new must be shorter than "n_data_max" since otherwise we cannot
    #    predict the label of all of the new data.
    if np.shape(X_new)[0] > n_data_max:
        raise ValueError('The number of new data needs to be less than "n_data_max".')
        
    # 6) Need to have one supervised method, and one only, provided.
    if supervised_method is None:
        raise ValueError('One supervised_method must be supplied to the function.')

    now_scores = kwargs.get('now_scores', None)
    
    #Initialize:
    supervised_indices = []
    indices_to_expert = []
    #n_send = -1
    
    #Next new batch size:
    myB = np.shape(X_new)[0]
    
    #Calculate n_top and n_active, the number of items in each loop that will be sent to
    #the expert for anomaly prediction and active learning, respectively.
    n_top = int(np.ceil(pc_top*n_send))
    n_active = n_send - n_top
    
    #Concatenate the old data to the new data:
    X_curr = np.concatenate([X_old,X_new],axis=0)
    
    # Send at most the last n_data_max rows to the methods for scoring (in order to keep
    # run time reasonable):
    if np.shape(X_curr)[0] > n_data_max:
        X_curr = X_curr[-n_data_max:,:]

    
    #######################################################################################
    # CALCULATING SCORES ##################################################################
    #######################################################################################
    
    # We require there to be at least n_data_min data points before calculating scores
    if np.shape(X_curr)[0] >= n_data_min:

                    
        # Define array to stock all current scores. Note that if X_curr contains data from the
        # original X_old, scores will be calculated ANEW here BUT if there had been sufficient 
        # labeled data in X_old originally to start to fill in "all_labeled_scores" previously,
        # WE DO NOT UPDATE "all_labeled_scores" based on the recalculation of scores here.
        
        # Calculate scores for all current data and all models: 
        all_scores = np.empty([np.shape(X_curr)[0],len(models.items())])

        # We run through the unsupervised methods, one by one, to output scores:
        
        
        ################# KEEP JUST IN CASE !!!! ###################################
        #formalizer = PandasFormalizer(data_df=X_curr, dataframe_type="synchronous")
        #results = {}
        #base_query = formalizer.default_query()
        #X = formalizer.formalize(**base_query)
        #base_query["target_period"] = (data.index[0])
        ############################################################################
        ############################################################################

        if curr_method == "RandomScore":
                all_scores = now_scores
        else:
            for i, (name, model) in enumerate(models.items()):
                model.fit(X_curr)
                y_score = model.score_samples(X_curr)
                y_score.dtype = np.float64
                all_scores[:,i] = y_score.squeeze()
        
                # For simplicity, we only keep the scores corresponding to new data:
                all_scores = all_scores[-myB:,:]

        
        #print('all scores top:',all_scores[:10,])
        #print('Length of all_scores is: ',np.shape(all_scores)[0])
        
        # Supervised learning interlude: if we have sufficient labeled data 
        # (either from the initialization step or from PREVIOUS loops of the
        # present function), AND we have at least five (for example) labeled item from each class, 
        # we run the machine learning method chosen. It is likely that the set
        # of allowed methods here must be able to use "predict" in Python.
        
        
        if len(Y_lab) >= min_n_labeled and sum([1 for i in range(len(Y_lab)) if Y_lab[i]==0]) >= min_n_nom and sum([1 for i in range(len(Y_lab)) if Y_lab[i]==1]) >= min_n_anom:
                
            # We find the method in a list and run it:
            
            #########################################################################################
            
            # 1) Random forest classifier:
            if supervised_method == 'RandomForestClassifier':
                #We want to use the classifier set-up first since we can weigh the classes differently.
                #But afterwards we really want to treat it as regression in order to propose candidates
                #to an expert.

                r0 = 1-tau_exp  # Proportion of class 0 expected
                weights = {0: r0, 1: 1 - r0}
                RFC = RandomForestClassifier(class_weight=weights)
                learned_model = RFC.fit(all_labeled_scores, Y_lab)
                # Predicted probabilities for the positive class
                new_preds = learned_model.predict_proba(all_scores)[:, 1]
                #print('new_preds',new_preds)
                
                #print('new_preds',new_preds)

            if supervised_method == 'LogisticRegression':
                #r0 = 1-tau_exp  # Proportion of class 0 expected
                #weights = {0: r0, 1: 1 - r0}          
                #LRC = LogisticRegression(class_weight=weights)
                #LRC = LogisticRegression(class_weight='balanced',penalty='l2')
                LRC = LogisticRegression(class_weight='balanced')
                
                learned_model = LRC.fit(all_labeled_scores, Y_lab)
                # Predicted probabilities for the positive class
                new_preds = learned_model.predict_proba(all_scores)[:, 1]
                
                

            
            # We will then use this learned model to predict the label of all of the data in X_curr.
            # corresponding to new data only:
            #new_preds = learned_model.predict(all_scores)
            #print(new_preds)

            #n_new_pred_anom = sum([r==1 for r in new_preds])
            
            # How good is this learned model 
            
            # Given that these predictions are set up so that the higher the prediction, the more
            # we suspect a data point is an anomaly, it makes sense to concentrate on sending the 
            # "highest" predictions to an expert who is looking for anomalies. There are a variety
            # of ways this could be done. For now, we hard code one method, which is to send the
            # top pc_pred fraction of predictions to the expert (or at a minimum, 1 prediction)
            
            #n_send = int(max(np.floor(pc_pred*myB),1))
            
            #Or we send all predicted anomalies:
            #n_send = n_new_pred_anom
            
            # We have to then sort "new_preds" and get the indices of the n_send largest:
            
            sorted_indices = np.argsort(new_preds)
            #print('sorted indices',sorted_indices)
            if n_top > 0:
                supervised_indices = sorted_indices[-n_top:].tolist()
            else:
                supervised_indices = []
            
            #sorted_indices = sorted(range(len(new_preds)), key=lambda k: new_preds[k])
            #supervised_indices = sorted_indices[-n_top:] 
            #print('Supervised_indices:',supervised_indices)
            #Print the corresponding values of x in the 1d case:
            #X_end = X[-myB:,:]
            #print('The x values predicted to be anomalies: ',X_end[supervised_indices])
        
        else: 
            learned_model = None


        #######################################################################################
        # ACTIVE LEARNING #####################################################################
        #######################################################################################

        # Here we implement a module that performs active learning. There are an enormous 
        # number of ways to go about this, whether it be which strategy to use, but also from
        # which pool of data do we look for candidates.

        # We shall do active learning by wrapping around the modAL Python package, which
        # implements active learning by itself (mostly) wrapping around methods in ScikitLearn.

        # For the moment, we use only the data in
        # X_new as potential data to give to the expert, given the labeled data we already have
        # from previous loops (if any). In the next phase of code development we can perhaps
        # also include ALL previous unlabeled data in this section.

        # First we need to know if there is already some labeled data or not. If there is
        # so far NO labeled data, or only data labeled from one class, we propose to use 
        # random sampling. If there is already labeled data from both classes, we shall
        # use the 

        active_indices = []
        curr_all_scores = all_scores

        if len(Y_lab) > min_n_labeled and sum([1 for i in range(len(Y_lab)) if Y_lab[i]==0]) >= min_n_nom and sum([1 for i in range(len(Y_lab)) if Y_lab[i]==1]) >= min_n_anom:
            
            
            curr_all_labeled_scores = all_labeled_scores
            curr_Y_lab = Y_lab
            
            # Set up the chosen active learning strategy (default in modAL is uncertainty sampling)
            if supervised_method == 'RandomForestClassifier':
                r0 = 1-tau_exp  # Proportion of class 0 expected
                weights = {0: r0, 1: 1 - r0}
                learner = ActiveLearner(
                #estimator=RandomForestClassifier(class_weight='balanced'),
                estimator=RandomForestClassifier(class_weight=weights),
                #query_strategy=uncertainty_sampling,
                query_strategy = margin_sampling,
                #query_strategy = entropy_sampling,
                X_training=curr_all_labeled_scores, y_training=curr_Y_lab
                )
                
            if supervised_method == 'LogisticRegression':
                #r0 = 1-tau_exp  # Proportion of class 0 expected
                #weights = {0: r0, 1: 1 - r0}
                learner = ActiveLearner(
                #estimator=LogisticRegression(class_weight=weights),
                #estimator=LogisticRegression(class_weight='balanced',penalty='l2'),
                estimator=LogisticRegression(class_weight='balanced'),
                #query_strategy=uncertainty_sampling,
                query_strategy = margin_sampling,
                #query_strategy = entropy_sampling,
                X_training=curr_all_labeled_scores, y_training=curr_Y_lab
                )
                
            
                
            #Since we want to do active learning using the modAL package, unfortunately it only 
            #outputs one candidate at a time, so there is a whole lot of work to do to make sure
            #nothing stupid happens with indices as we iteratively output one candidate at a time
            #until we get to n_active.

            true_indices = list(range(0,np.shape(curr_all_scores)[0]))

            i = 0
            while i < n_active:    
                          
                query_idx, query_inst = learner.query(curr_all_scores)
                #print('query_idx = ',query_idx)
                #print('query_inst = ',query_inst)
                query_idx = int(query_idx[0])
                

                # Get the actual index:
                actual_index = true_indices[query_idx]
                #print('Actual index:',actual_index)

                # See whether it was already in supervised_indices and if so, skip it.
                if actual_index in supervised_indices:
                    #print('It was a repeat')
                    #Remove this index from true_indices:
                    del true_indices[query_idx]
                    # Remove this guy from the data and loop back:
                    curr_all_scores = np.delete(curr_all_scores, query_idx, axis=0)
                    continue
                    
                #else:
                    # Append new index to list:               
                active_indices.append(true_indices[query_idx])

                
                #Remove this index from true_indices:
                del true_indices[query_idx]
                
                # Remove this guy from the data and loop back:
                curr_all_scores = np.delete(curr_all_scores, query_idx, axis=0)
                #del curr_Y_lab[query_idx]

                i += 1


        else:
            print('The condition was not satisfied!')
            # There is no (or not enough) labeled data, so we do random sampling:
            my_random_permute = np.random.permutation(np.shape(curr_all_scores)[0])
            #learner = ActiveLearner(
            #estimator=RandomForestClassifier(class_weight='balanced'),    #irrelevant
            #query_strategy=random_sampling,
            #X_training=X_curr, y_training=np.ones((np.shape(X_curr)[0],1))     #irrelevant
            #)

            # query for labels
            #query_idx, query_inst = learner.query(curr_all_scores)
            # Make sure that the indices of the selected data are integers and turn this into a list:
            
            active_indices = [int(my_random_permute[i]) for i in range(n_active)]


        #print('active_indices:',type(active_indices))
        #print('supervised_indices:',type(supervised_indices))
        # So now we have both data from the supervised prediction, AND data from the active
        # learning, to give to the expert for labeling. We need to merge these two lists of 
        # indices as they may have overlaps
        
        
        
        indices_to_expert = supervised_indices + active_indices
        #print('superv_ind_merged_with_activ_ind:',indices_to_expert)
        indices_to_expert = sorted(set(indices_to_expert))
    
    
    
        #######################################################################################
        # UPDATING PARAMETERS #################################################################
        #######################################################################################
    
        
        
        
        # Update all_labeled_scores:
        if np.shape(all_labeled_scores) != (0,0): 
            all_labeled_scores = np.concatenate([all_labeled_scores,all_scores[indices_to_expert,:]],axis=0)


        #print('merged indices sent to expert:',indices_to_expert)

    
        # Update X_lab:
        if np.shape(X_lab) != (0,0):
            X_new_labeled = X_new[indices_to_expert,:]
            X_lab = np.concatenate([X_lab,X_new_labeled],axis=0)
        else:
            X_lab = X_new[indices_to_expert,:]
    
    # Update X_old:
    X_old = np.concatenate([X_old,X_new],axis=0)
    
    # Retain at most n_data_max points in X_old:
        
    if np.shape(X_old)[0] > n_data_max:
        X_old = X_old[-n_data_max:,:]
        
    
    
    return X_old, X_lab, all_labeled_scores, indices_to_expert, learned_model, supervised_indices 

###################################################################################################
###################################################################################################


def random_sampling(classifier, X_pool, pc_active):
    n_samples = np.shape(X_pool)[0]
    n_return = max(int(np.floor(pc_active*n_samples)),1)
    query_idx = np.random.choice(range(n_samples),size=n_return,replace=False)
    return query_idx, X_pool[query_idx] 


###################################################################################################
###################################################################################################


def new_custom_loss_2(X_lab, Y_lab, q_tau_tm1, all_labeled_scores, model, X_so_far, mylambda=1, b=0.5):
    
    """
    Optimized version of GLAD's custom loss using vectorized operations.
    """

    # Convert Y_lab to {-1, 1}
    Y_lab_tensor = tf.convert_to_tensor(2 * np.array(Y_lab) - 1, dtype=tf.float32)

    # Convert all_labeled_scores to tensor
    scores_tensor = tf.convert_to_tensor(all_labeled_scores, dtype=tf.float32)

    # Compute all weights in one forward pass
    w_tensor = model(X_lab)  # Shape: (n_labeled, M)

    # Compute weighted scores (batch matmul equivalent to inner product row-wise)
    weighted_scores = tf.reduce_sum(w_tensor * scores_tensor, axis=1)  # Shape: (n_labeled,)

    # Compute first term using vectorized hinge-like loss
    hinge_losses = tf.maximum(0.0, Y_lab_tensor * (q_tau_tm1 - weighted_scores))
    first_term = tf.reduce_mean(hinge_losses)

    # Second term: cross-entropy on X_so_far
    outputs = model(X_so_far)  # Shape: (n_so_far, M)
    binary_loss_fn = custom_binary_crossentropy_loss(b, mylambda)
    second_term = tf.reduce_mean(binary_loss_fn(None, outputs)) * mylambda

    # Total loss
    return first_term + second_term

###################################################################################################
###################################################################################################

class RandomScore:
    def __init__(self):
        pass
    
    def fit(self, X):
        # No actual fitting logic for RandomScore; just a placeholder
        self.X = X
    
    def score_samples(self, X):
        # Assign a random score between 0 and 1 for each data point
        return np.random.rand(X.shape[0])


###################################################################################################
###################################################################################################

def build_random_score_models(M=50):
    """
    Build a dictionary of M RandomScore models, each with a unique name.

    Parameters:
    - M: Number of RandomScore models to generate (default=50).

    Returns:
    - models: Dictionary of RandomScore models, each with a unique name.
    """
    models = {}
    
    # Create M RandomScore models and add them to the dictionary
    for m_idx in range(M):
        model = RandomScore()
        models[f'model_{m_idx + 1}'] = model
    
    return models
    

###################################################################################################
###################################################################################################


class GaussianScoreModel:
    def __init__(self, mean=0, std=1):
        """
        Initialize the GaussianScoreModel with the mean and standard deviation of the Gaussian distribution.

        Parameters:
        - mean: Mean of the Gaussian distribution.
        - std: Standard deviation of the Gaussian distribution.
        """
        self.mean = mean
        self.std = std

    def fit(self, X):
        # Placeholder fit method (no action needed for this model).
        self.X = X

    def score_samples(self, X):
        """
        Generate scores for each row of X from a Gaussian distribution.

        Parameters:
        - X: Input data array.

        Returns:
        - scores: A 1D array of scores for each row in X.
        """
        return np.random.normal(loc=self.mean, scale=self.std, size=X.shape[0])
        

###################################################################################################
###################################################################################################


class MixtureGaussianScoreModel:
    def __init__(self, tau=0.01, nominal_mean=0, nominal_std=1, anomaly_mean=2, anomaly_std=0.1):
        """
        Initialize the MixtureGaussianScoreModel with parameters for the nominal and anomaly distributions.

        Parameters:
        - tau: Probability of sampling from the anomaly distribution.
        - nominal_mean: Mean of the nominal Gaussian distribution.
        - nominal_std: Standard deviation of the nominal Gaussian distribution.
        - anomaly_mean: Mean of the anomaly Gaussian distribution.
        - anomaly_std: Standard deviation of the anomaly Gaussian distribution.
        """
        self.tau = tau
        self.nominal_mean = nominal_mean
        self.nominal_std = nominal_std
        self.anomaly_mean = anomaly_mean
        self.anomaly_std = anomaly_std

    def fit(self, X):
        # Placeholder fit method (no action needed for this model).
        self.X = X

    def score_samples(self, X):
        """
        Generate scores for each row of X from either the nominal or anomaly distribution.

        Parameters:
        - X: Input data array.

        Returns:
        - scores: A 1D array of scores for each row in X.
        - labels: A 1D array of labels indicating the source of each score (0 = nominal, 1 = anomaly).
        """
        num_samples = X.shape[0]
        scores = np.zeros(num_samples)
        labels = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            if np.random.rand() < self.tau:
                # Generate score from the anomaly distribution
                scores[i] = np.random.normal(loc=self.anomaly_mean, scale=self.anomaly_std)
                labels[i] = 1
            else:
                # Generate score from the nominal distribution
                scores[i] = np.random.normal(loc=self.nominal_mean, scale=self.nominal_std)
                labels[i] = 0

        return scores, labels


###################################################################################################
###################################################################################################


def build_custom_score_models(M=50, tau=0.01, nominal_params=(0, 1), anomaly_params=(2, 1)):
    """
    Build a dictionary of M models, where M-1 models generate scores from a single Gaussian distribution,
    and the last model generates scores from a mixture of two Gaussians.

    Parameters:
    - M: Total number of models to generate.
    - tau: Probability of sampling from the anomaly distribution in the final model.
    - nominal_params: Tuple (mean, std) for the nominal Gaussian distribution.
    - anomaly_params: Tuple (mean, std) for the anomaly Gaussian distribution.

    Returns:
    - models: Dictionary of models.
    """
    models = {}

    # Create M-1 GaussianScoreModel instances
    for m_idx in range(M - 1):
        mean, std = nominal_params
        model = GaussianScoreModel(mean=mean, std=std)
        models[f'model_{m_idx + 1}'] = model

    # Create the final MixtureGaussianScoreModel
    nominal_mean, nominal_std = nominal_params
    anomaly_mean, anomaly_std = anomaly_params
    final_model = MixtureGaussianScoreModel(
        tau=tau,
        nominal_mean=nominal_mean,
        nominal_std=nominal_std,
        anomaly_mean=anomaly_mean,
        anomaly_std=anomaly_std
    )
    models[f'model_{M}'] = final_model

    return models


###################################################################################################
###################################################################################################


class ExtendedMixtureGaussianScoreModel:
    def __init__(self, tau=0.01, nominal_mean=0, nominal_std=1, 
                 anomaly1_mean=2, anomaly1_std=0.1, 
                 anomaly2_mean=-2, anomaly2_std=0.1):
        """
        Initialize the ExtendedMixtureGaussianScoreModel with parameters for the nominal and two anomaly distributions.

        Parameters:
        - tau: Probability of sampling from the anomaly distributions (split equally between two anomalies).
        - nominal_mean: Mean of the nominal Gaussian distribution.
        - nominal_std: Standard deviation of the nominal Gaussian distribution.
        - anomaly1_mean: Mean of the first anomaly Gaussian distribution.
        - anomaly1_std: Standard deviation of the first anomaly Gaussian distribution.
        - anomaly2_mean: Mean of the second anomaly Gaussian distribution.
        - anomaly2_std: Standard deviation of the second anomaly Gaussian distribution.
        """
        self.tau = tau
        self.nominal_mean = nominal_mean
        self.nominal_std = nominal_std
        self.anomaly1_mean = anomaly1_mean
        self.anomaly1_std = anomaly1_std
        self.anomaly2_mean = anomaly2_mean
        self.anomaly2_std = anomaly2_std

    def fit(self, X):
        # Placeholder fit method (no action needed for this model).
        self.X = X

    def score_samples(self, X):
        """
        Generate scores for each row of X from the nominal or one of the anomaly distributions.

        Parameters:
        - X: Input data array.

        Returns:
        - scores: A 1D array of scores for each row in X.
        - labels: A 1D array of labels indicating the source of each score 
                  (0 = nominal, 1 = anomaly1, 2 = anomaly2).
        """
        num_samples = X.shape[0]
        scores = np.zeros(num_samples)
        labels = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            rand = np.random.rand()
            if rand < self.tau / 2:
                # Sample from anomaly1 distribution
                scores[i] = np.random.normal(loc=self.anomaly1_mean, scale=self.anomaly1_std)
                labels[i] = 1
            elif rand < self.tau:
                # Sample from anomaly2 distribution
                scores[i] = np.random.normal(loc=self.anomaly2_mean, scale=self.anomaly2_std)
                labels[i] = 1
            else:
                # Sample from nominal distribution
                scores[i] = np.random.normal(loc=self.nominal_mean, scale=self.nominal_std)
                labels[i] = 0

        return scores, labels
        

###################################################################################################
###################################################################################################


def build_custom_score_models2(
    M=50, 
    tau=0.01, 
    nominal_params=(0, 1), 
    anomaly_params_list=[(2, 1), (3, 0.5)]
):
    """
    Build a dictionary of M models, where M-1 models generate scores from a single Gaussian distribution,
    and the last model generates scores from a mixture of one nominal and two anomaly distributions.

    Parameters:
    - M: Total number of models to generate.
    - tau: Probability of sampling from the anomaly distributions in the final model.
    - nominal_params: Tuple (mean, std) for the nominal Gaussian distribution.
    - anomaly_params_list: List of tuples [(mean, std), ...] for the anomaly Gaussian distributions.

    Returns:
    - models: Dictionary of models.
    """
    models = {}

    # Create M-1 GaussianScoreModel instances
    for m_idx in range(M - 1):
        mean, std = nominal_params
        model = GaussianScoreModel(mean=mean, std=std)
        models[f'model_{m_idx + 1}'] = model

    # Create the final ExtendedMixtureGaussianScoreModel
    nominal_mean, nominal_std = nominal_params
    anomaly_params_1 = anomaly_params_list[0]
    anomaly_params_2 = anomaly_params_list[1]
    final_model = ExtendedMixtureGaussianScoreModel(
        tau=tau,
        nominal_mean=nominal_mean,
        nominal_std=nominal_std,
        anomaly1_mean=anomaly_params_1[0],
        anomaly1_std=anomaly_params_1[1],
        anomaly2_mean=anomaly_params_2[0],
        anomaly2_std=anomaly_params_2[1]
    )
    models[f'model_{M}'] = final_model

    return models


###################################################################################################
###################################################################################################


def generate_time_series(my_tau, d, n, 
                         mean_G1=None, mean_G2=None, mean_G3=None, 
                         cov_G1=None, cov_G2=None, cov_G3=None, 
                         c=1,c_anom = 1):
    """
    Generates a time series with values drawn from three multivariate Gaussians G1, G2, and G3.
    
    Parameters:
    - tau: Probability of selecting G2.
    - d: Data dimensionality.
    - n: Number of data points to generate.
    - mean_G1, mean_G2, mean_G3: Means of the Gaussians (default: [1,...,1], [2,...,2], [3,...,3]).
    - cov_G1, cov_G2, cov_G3: Covariance matrices (default: c * Identity matrix).
    - c: Scaling factor for identity covariance matrices (default: 1).

    Returns:
    - time_series: Generated time series (n x d array).
    - Y: Labels indicating whether each point came from G2 (1) or from G1/G3 (0).
    """
    # Default mean values
    if mean_G1 is None:
        mean_G1 = np.ones(d)
    if mean_G2 is None:
        mean_G2 = np.full(d, 2)
    if mean_G3 is None:
        mean_G3 = np.full(d, 3)
    
    # Default covariance matrices
    if cov_G1 is None:
        cov_G1 = c * np.eye(d)
    if cov_G2 is None:
        cov_G2 = c_anom * np.eye(d)
    if cov_G3 is None:
        cov_G3 = c * np.eye(d)
    
    # Initialize time series storage
    time_series = np.zeros((n, d))
    
    # Initialize Y labels
    Y = np.empty(n, dtype=int)

    # First value is generated independently
    probabilities = [(1 - my_tau) / 2, my_tau, (1 - my_tau) / 2]
    choice = np.random.choice([1, 2, 3], p=probabilities)

    if choice == 1:
        temp_value = np.random.multivariate_normal(mean_G1, cov_G1)
        Y[0] = 0
    elif choice == 2:
        temp_value = np.random.multivariate_normal(mean_G2, cov_G2)
        Y[0] = 1
    else:
        temp_value = np.random.multivariate_normal(mean_G3, cov_G3)
        Y[0] = 0

    if np.random.rand() < 0.5:
        temp_value *= -1  # Flip sign with probability 0.5

    time_series[0] = temp_value  # First value

    # Generate the rest of the series
    for i in range(1, n):
        choice = np.random.choice([1, 2, 3], p=probabilities)

        if choice == 1:
            temp_value = np.random.multivariate_normal(mean_G1, cov_G1)
            Y[i] = 0
        elif choice == 2:
            temp_value = np.random.multivariate_normal(mean_G2, cov_G2)
            Y[i] = 1
        else:
            temp_value = np.random.multivariate_normal(mean_G3, cov_G3)
            Y[i] = 0

        if np.random.rand() < 0.5:
            temp_value *= -1  # Flip sign with probability 0.5


        time_series[i] = time_series[i - 1] + temp_value  # Cumulative sum

    return time_series, Y

def plot_time_series(X, Y):
    n = len(Y)
    
    plt.figure(figsize=(12, 6))  # Set figure size
    #plt.plot(range(n), X, linestyle='-', color='gray', alpha=0.5, label="Time Series")  # Plot the time series

    # Scatter plot with colors based on Y values
    plt.scatter(range(n), X, c=['blue' if y == 0 else 'red' for y in Y], s=2)

    # Labels and title
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Generated Time Series")
    plt.legend()
    plt.grid(True)

    plt.show()


###################################################################################################
###################################################################################################

class EuclideanDifferenceAnomalyDetector:
    def __init__(self):
        """
        Initializes the Euclidean Difference Anomaly Detector.
        This model calculates the Euclidean distance between consecutive rows of the data.
        """
        pass
    
    def fit(self, X):
        """
        Fits the model to the data (although no actual fitting is necessary here).
        This is just for consistency with the framework.
        
        Parameters:
        - X: The data to fit the model to.
        """
        # In this case, there's no fitting process required, so we'll just store X if needed
        self.X = X  # Optional: We store X for future use if needed.
    
    def score_samples(self, X):
        """
        Computes the Euclidean distance between consecutive rows and returns the scores.
        
        Parameters:
        - X: The data to compute anomaly scores for (shape: [n_samples, n_features]).
        
        Returns:
        - scores: A NumPy array of scores, where each value is the Euclidean distance 
                  from the previous row.
        """
        # Initialize a list to hold the scores
        scores = np.zeros(X.shape[0])
        
        # Calculate the Euclidean distances between consecutive rows
        for i in range(1, X.shape[0]):
            # Euclidean distance between row i and row (i-1)
            dist = np.linalg.norm(X[i] - X[i-1])
            scores[i] = dist
        
        # The first row score is set to the same as the second row score
        scores[0] = scores[1]
        
        
        return scores


###################################################################################################
###################################################################################################


def Create_Anomaly_Models(d, n_LODA_models=0, additional_models=None):
    """
    Creates a fixed number of LODA models and user-specified anomaly detection models.
    
    Parameters:
    - d: Data dimensionality.
    - n_LODA_models: The number of LODA models to generate.
    - additional_models: Dictionary of {model_name: uninitialized model instance}.

    Returns:
    - models: Dictionary containing all models.
    """
    
    # Initialize models dictionary
    models = {}
    
    # Loop to create n_LODA_models
    for i in range(n_LODA_models):
        # Increment LODA_index to generate unique model names (Loda_1, Loda_2, ..., Loda_M)
        current_LODA_index = i + 1
        
        # Generate a single LODA model and add it to the models dictionary
        new_model, proj_vectors = LODA_OAT(M=1, d=d, models=None, LODA_index=current_LODA_index)
        
        # Add the newly generated models to the models dictionary
        models.update(new_model)
    
    # Add user-specified models (e.g., IsolationForest, OneClassSVM)
    if additional_models:
        for model_name, model_instance in additional_models.items():
            models[model_name] = model_instance

    return models


###################################################################################################
###################################################################################################


def Compute_Model_Scores(X, models):
    """
    Compute anomaly scores for each model in the models dictionary.
    
    Parameters:
    - X: The data to compute the scores for (shape: [n_samples, n_features]).
    - models: Dictionary of models to compute scores from.
    
    Returns:
    - scores_array: A NumPy array where each row corresponds to a data point
                     and each column corresponds to a model's score for that data point.
    """
    tiny_eps = .0000000001
    # Initialize the scores array with the correct shape: n_samples x n_models
    n_samples = X.shape[0]
    n_models = len(models)
    scores_array = np.zeros((n_samples, n_models))
    
    # Loop over each model and calculate the scores
    for i, (model_name, model) in enumerate(models.items()):
        
        # Get the raw scores and flatten them
        if isinstance(model, OneClassSVM):
            # Use decision_function() for OneClassSVM to compute anomaly scores
            model.fit(X)
            scores = model.decision_function(X).ravel()
            #The above has anomalies as smaller values, and can be negative. Correction:
            scores = -scores
            minOne = np.min(scores)
            scores = scores - minOne + tiny_eps
            #max_value = np.max(scores)
            #scores = max_value - scores  # Invert the scores: Higher scores for anomalies
        elif isinstance(model, IsolationForest):
            model.fit(X)
            # Use decision_function() for OneClassSVM to compute anomaly scores
            scores = model.decision_function(X).ravel()
            #The above has anomalies as smaller values, and can be negative. Correction:
            scores = -scores
            minIso = np.min(scores)
            scores = scores - minIso + tiny_eps
            #scores = -decision_fn  # Invert the scores: Higher scores for anomalies

        elif isinstance(model, LocalOutlierFactor):
            model.fit(X)
            # Use decision_function() for OneClassSVM to compute anomaly scores
            scores = model.decision_function(X).ravel()
            #The above has anomalies as smaller values, and can be negative. Correction:
            scores = -scores
            minLOF = np.min(scores)
            scores = scores - minLOF + tiny_eps
            #scores = -decision_fn  # Invert the scores: Higher scores for anomalies

        
        else:
            # Use score_samples() for other models 
            model.fit(X)
            scores = model.score_samples(X).ravel()  # Flatten the scores to a 1D array
            #print(scores)
            
            # For IsolationForest, invert the scores for higher scores for anomalies
            #if isinstance(model, IsolationForest):
                #scores = -scores
        
        scores_array[:, i] = scores
        
    return scores_array


###################################################################################################
###################################################################################################


###################################################################################################
###################################################################################################
