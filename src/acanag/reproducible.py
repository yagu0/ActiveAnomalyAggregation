## A bunch of functions required only to run the reproducible Jupyter Notebook 
## for the paper's plot outputs.

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner

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
                    
                    for i, (name, model) in enumerate(models.items()):
                        model.fit(X_old)
                        y_score = model.score_samples(X_old)
                        y_score.dtype = np.float64
                        all_scores[:,i] = y_score.squeeze()

                    
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


def random_sampling(classifier, X_pool, pc_active):
    n_samples = np.shape(X_pool)[0]
    n_return = max(int(np.floor(pc_active*n_samples)),1)
    query_idx = np.random.choice(range(n_samples),size=n_return,replace=False)
    return query_idx, X_pool[query_idx] 


###################################################################################################
###################################################################################################






###################################################################################################
###################################################################################################






###################################################################################################
###################################################################################################






###################################################################################################
###################################################################################################
