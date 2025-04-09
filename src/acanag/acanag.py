import numpy as np
from modAL.uncertainty import margin_sampling
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner

def init_super_sad(X_old = None,Y_old = None,data_dim = None,n_data_min = 100, models=None):

    """
    This function runs before the main super_sad() function. It treats the existence
    or absence of data provided from the past (prior knowledge via possibly labeled data)
    and initializes X_lab, Y_lab, and all_labeled_scores.

    Parameters:
       X_old: either not included (no old data) or a numpy array of shape
               (number of old data) x (number of data dimensions)
       Y_old: either not included (no old data) or a list of items with values 0, 1, or nan.
               Note that if X_old is a non-empty array, we require Y_old to exist, even if
               it is filled completely with nan (no labeled data)
       data_dim: a positive integer corresponding to the dimension of the data that will be seen
                 in the super_sad() function and should also be the data dimension of X_old if
                 it is non-empty.
       n_data_min: minimum number of data points required before calculating any anomaly scores
                    (default = 100. We do not recommend less than this)
       models: a dictionary my_models of (unsupervised) anomaly detection models.


    Outputs:
        X_old: Either the X_old that was input, or an empty array with 0 rows and data_dim columns
        X_lab: the (possibly empty) array containing the subset of old data (should it exist)
                which is already labeled from the past
        Y_lab: the (possibly empty) list containing the labels 0 (for nominal) or 1 (for anomaly)
                corresponding to X_lab's data points
        all_labeled_scores: the (possibly empty) array containing the M-dimensional anomaly scores
                corresponding to the data-points in X_lab, where M is the number of anomaly
                detection models chosen by the user.
    """

    ##############################################################################
    # Error detection

    # Check that data_dim is a positive integer
    if not isinstance(data_dim, int) or data_dim <= 0:
        raise ValueError("data_dim must be a positive integer.")

    #Dealing with X_old
    if X_old is None:
        # If X_old is None, initialize it as an empty numpy array with 0 rows and data_dim columns
        X_old = np.empty((0, data_dim))
    else:
        # If X_old is not None, check that it is a numpy array and that it has data_dim columns
        if not isinstance(X_old, np.ndarray):
            raise ValueError("X_old must be a numpy array.")

        if X_old.shape[1] != data_dim:
            raise ValueError(f"X_old must have {data_dim} columns, but it has {X_old.shape[1]} columns.")

    #Dealing with Y_old
    if Y_old is None:
        Y_old = []  # If Y_old is None, initialize it as an empty list
    else:
        if not isinstance(Y_old, list):
            raise ValueError("Y_old must be a list.")

        # Check thats Y_old has the same number of entries as the number of rows of X_old
        if len(Y_old) != X_old.shape[0]:
            raise ValueError(f"Y_old must have the same number of entries as X_old has rows. Expected {X_old.shape[0]} entries, but got {len(Y_old)}.")

        # Only check for 0, 1, or nan if Y_old is not empty
        if len(Y_old) > 0:
            for y in Y_old:
                if y not in [0, 1] and not np.isnan(y):  # Check if the entry is not 0, 1, or NaN
                    raise ValueError(f"Y_old entries must be 0, 1, or NaN. Found an invalid entry: {y}")

    # Check if n_data_min is an integer
    if not isinstance(n_data_min, int):
        raise ValueError(f"n_data_min must be an integer. Got {type(n_data_min)} instead.")

    # Check if n_data_min is positive
    if n_data_min <= 0:
        raise ValueError(f"n_data_min must be a positive integer. Got {n_data_min}.")

    # Warning if n_data_min is less than 100
    if n_data_min < 100:
        warnings.warn(f"Warning: n_data_min is less than 100. It is set to {n_data_min}. Consider increasing it for more reliable results.", UserWarning)

    # Check if models is not None and is a dictionary
    if not isinstance(models, dict):
        raise ValueError("models must provided, and it must be a dictionary.")

    # Check if models is not empty
    if len(models) == 0:
        raise ValueError("models must contain at least one dictionary item.")

    ########################################################################################

    #If there is no old data:
    if np.shape(X_old)[0] == 0:

        X_lab = np.empty([0,data_dim])
        Y_lab = []
        all_labeled_scores = np.empty([0,len(models.items())])

    #Else there is old data:
    else:
        #deal with all cases: (no labels, some labels, fully labeled):
        which_lab = [i for i in range(len(Y_old)) if not np.isnan(Y_old[i])]

        #If none of the old data is labeled:
        if len(which_lab) == 0:
            X_lab = np.empty([0,data_dim])
            Y_lab = []
            all_labeled_scores = np.empty([0,len(models.items())])

        #Else at least some of it is labeled, and so:
        else:
            X_lab = X_old[which_lab,:]
            Y_lab = [Y_old[j] for j in which_lab]

            # If there is at least one item with label 0 OR one item with label 1,
            # we can potentially start to calculate scores for each available model:
            if sum([1 for i in range(len(Y_lab)) if Y_lab[i]==0]) > 0 or sum([1 for i in range(len(Y_lab)) if Y_lab[i]==1]) > 0:

                # Since there are labeled data from each class, now we want to check that we have
                # enough data OVERALL for calculating meaningful anomaly socres. If we do not,
                # then no scores are calculated by this initialization function. By this we mean
                # that there are many anomaly detectors which require a large enough bag or batch
                # to output meaningful "stable" scores (e.g., Isolation Forest on a dataset of size 10
                # risks being very unstable, that is, the same point could have wildly different scores
                # depending on the other nine points' values, something which is less likely to
                # happen if there are say 500 points):

                if np.shape(X_old)[0] >= n_data_min:

                    # Define arrays to stock all scores:
                    all_scores = np.empty([np.shape(X_old)[0],len(models.items())])

                    # We run through the unsupervised models, one by one, to output scores:
                    for i, (name, model) in enumerate(models.items()):
                        model.fit(X_old)
                        y_score = model.score_samples(X_old)
                        y_score.dtype = np.float64
                        all_scores[:,i] = y_score.squeeze()

                    # We then extract the subset of the array with the scores for the labeled data:
                    all_labeled_scores = np.take(all_scores, which_lab, 0)

                else:
                    all_labeled_scores = np.empty([0,len(models.items())])

            else:
                all_labeled_scores = np.empty([0,len(models.items())])


    return X_old, X_lab, Y_lab, all_labeled_scores

def super_sad(X_new = None, X_old = None, X_lab = None, Y_lab = None, all_labeled_scores = None, models=None, supervised_model={'class': RandomForestClassifier, 'params': {'class_weight': {0: 0.999, 1: 0.001}}}, query_strategy = margin_sampling, n_data_min = 100, n_data_max = 2000, min_n_labeled = 5, n_send=2, pc_top = 0.5, min_n_nom=5, min_n_anom=1):

    """
    This function runs active anomaly detection on batch data. Each
    time it is called, it "eats" some new data, it runs all the models to get scores for each
    new data point, it uses active learning to propose some of these to an expert, and it
    predicts new anomalies using the current version of the supervised machine learning model.

    Remarks :
       1) The dictionary "models" of course needs to be IDENTICAL to the dictionary "models" in
           init_super_sad().

    Arguments:
       X_new: new numpy array batch of B data of size B x number of data dimensions
       X_old: numpy array of all (or most recent) past data.
       X_lab: numpy array of the current set of labeled data
       Y_lab: list of the current labels of the currently labeled data
       all_labeled_scores: current numpy array of all score vectors that have been labeled so far.
       models: a dictionary of (unsupervised) models that can output anomaly scores. This should
               be the same dictionary as in init_super_sad
       supervised_model: a choice of one supervised method from Scikit Learn that can do supervised
            learning on score vectors which have known labels (0 non-anomaly, 1 anomaly).
       query_strategy: a query strategy chosen from those that can be associated with the Python
                   modAL package: https://modal-python.readthedocs.io/en/latest/
                   (default: margin_sampling).
       n_data_min: minimum number of data points required before calculating any anomaly scores
            (default = 100). Note that your value for this should be the same value as
            used in init_super_sad().
       n_data_max: maximum number of most recent time points to send to each model (in order to
            keep run times faster).
       min_n_labeled: minimum number of labeled data required before running supervised learning.
       n_send: total number of unlabeled items that can be sent to the expert for any reason
            during any loop.
       pc_top: the percentage of the n_send items that are used to directly predict anomalies
           (as a fraction, e.g. 0.6) based on the current state of the supervised learner
       min_n_nom: minimum number of labeled nominals required before doing supervised and
            active learning
       min_n_anom: minimum number of labeled anomalies required before doing supervised and
            active learning

    Outputs:
        X_old: The at most n_data_max points most recent data points, including at a minimum all
            of the points from the last input variable X_new, if not also the most recent points
            previous to the current X_new.
        X_lab: updated array whose data rows correspond to all labeled data points so far.
        all_labeled_scores: an array with the same number of rows as X_lab, containing the scores
            corresponding to the data in the rows of X_lab
        indices_to_expert: the Python indices of the rows of X_new corresponding to those data points
            selected to give to the expert for labeling.
        learned_model: the current fitted supervised model trained on all_labeled_scores and Y_lab.
            Note that the model is only output in case it is needed elsewhere, such as when we are
            running simulations and we have a large external data set generated in the same way
            as the trial data, with known labels, that we want
            to independently test the learned_model on (e.g., for estimating the AUC).
        supervised_indices: the subset of indices_to_expert corresponding to the data points with
            the n_top highest predicted scores (and thus predicted anomalies). This is output just
            in case you want to make a distinction between the points sent to the expert for reasons
            of high predicted scores, and points sent to the expert from the active learning module
            (note however that sometimes there is an intersection between the two sets).
        active_indices: the subset of indices_to_expert corresponding to the data points selected
            by the active learner for labeling. This is output just
            in case you want to make a distinction between the points sent to the expert for reasons
            of high predicted scores, and points sent to the expert from the active learning module
            (note however that sometimes there is an intersection between the two sets).
    """

    ##################################################################################################
    #Error detection:

    # Ensure X_new is not None
    if X_new is None:
        raise ValueError("X_new must be given.")

    # Check if X_new is a numpy array
    if not isinstance(X_new, np.ndarray):
        raise ValueError("X_new must be a numpy array.")

    # Ensure X_new has at least one row and one column
    if X_new.shape[0] == 0:
        raise ValueError("X_new must have at least one row.")
    if X_new.shape[1] == 0:
        raise ValueError("X_new must have at least one column.")

    # Ensure X_old is provided
    if X_old is None:
        raise ValueError("X_old output from init_super_sad() must be provided.")

    # Check that X_new and X_old have the same number of columns
    if X_new.shape[1] != X_old.shape[1]:
        raise ValueError(f"X_new and X_old must have the same number of columns. X_new has {X_new.shape[1]} columns, but X_old has {X_old.shape[1]} columns.")

    # Ensure X_lab is provided
    if X_lab is None:
        raise ValueError("X_lab output from init_super_sad() must be provided.")

    # Ensure Y_lab is provided
    if Y_lab is None:
        raise ValueError("Y_lab output from init_super_sad() must be provided.")

    # Ensure all_labeled_scores is provided
    if all_labeled_scores is None:
        raise ValueError("all_labeled_scores output from init_super_sad() must be provided.")

    if models is None:
        raise ValueError('a models dictionary must be input, and it should be the same dictionary used as input to init_super_sad().')

    # Check if n_data_min is a positive integer
    if not isinstance(n_data_min, int) or n_data_min <= 0:
        raise ValueError(f"n_data_min must be a positive integer. Got {n_data_min}. Please also make sure it has the same value as it does in init_super_sad().")

    # Check if n_data_max is a positive integer
    if not isinstance(n_data_max, int) or n_data_max <= 0:
        raise ValueError(f"n_data_max must be a positive integer. Got {n_data_max}.")

    #n_data_min must be no bigger than n_data_max:
    if n_data_min > n_data_max:
        raise ValueError('"n_data_min" cannot be greater than "n_data_max".')

    #The new data X_new must have at most as many data points as n_data_max since
    #otherwise we cannot predict the label of all of the new data.
    if np.shape(X_new)[0] > n_data_max:
        raise ValueError('The number of new data needs to be less than or equal to "n_data_max".')

    # Ensure supervised_model is provided
    if supervised_model is None:
        raise ValueError("supervised_model must be provided as a dictionary containing 'class' and 'params'.")

    # Ensure supervised_model is a dictionary
    if not isinstance(supervised_model, dict):
        raise ValueError("supervised_model must be a dictionary containing 'class' and 'params'.")

    # Ensure required keys exist
    if 'class' not in supervised_model or 'params' not in supervised_model:
        raise ValueError("supervised_model dictionary must have both 'class' and 'params' keys.")

    # Extract classifier class and parameters
    classifier_class = supervised_model['class']
    classifier_params = supervised_model['params']

    # Ensure classifier_class is callable:
    if not callable(classifier_class):
        raise ValueError("The 'class' key in supervised_model must be a callable classifier class (e.g., RandomForestClassifier).")

    # Check if min_n_labeled is a positive integer and at least 2
    if not isinstance(min_n_labeled, int) or min_n_labeled < 2:
        raise ValueError(f"min_n_labeled must be a positive integer greater than or equal to 2. Got {min_n_labeled}.")

    # Check if n_send is a positive integer and at most equal to the number of rows in X_new
    if not isinstance(n_send, int) or n_send <= 0 or n_send > X_new.shape[0]:
        raise ValueError(f"n_send must be a positive integer less than or equal to the number of rows in X_new. Got {n_send}, but X_new has {X_new.shape[0]} rows.")

    # Check if pc_top is a fraction between 0 and 1, inclusive
    if not (0 <= pc_top <= 1):
        raise ValueError(f"pc_top must be a fraction between 0 and 1 (inclusive). Got {pc_top}.")

    # Check if min_n_nom is a positive integer and at least 1
    if not isinstance(min_n_nom, int) or min_n_nom < 1:
        raise ValueError(f"min_n_nom must be a positive integer greater than or equal to 1. Got {min_n_nom}.")

    # Check if min_n_anom is a positive integer and at least 1
    if not isinstance(min_n_anom, int) or min_n_anom < 1:
        raise ValueError(f"min_n_anom must be a positive integer greater than or equal to 1. Got {min_n_anom}.")

    # Ensure query_strategy is provided
    if query_strategy is None:
        raise ValueError("query_strategy must be specified as one of the options from the modAL package: https://modal-python.readthedocs.io/en/latest/.")

    # Checking to see if the user has retained the default supervised model dictionary by accident or design
    #if supervised_model == {'class': RandomForestClassifier, 'params': {'class_weight': {0: 0.999, 1: 0.001}}}:
    #    warnings.warn(
    #        "You are using the default supervised model (RandomForestClassifier with class_weight={0: 0.999, 1: 0.001}). "
    #        "Please replace it if you want to use a different model.",
    #        UserWarning
    #    )

    ######################################################################################

    #Initialize:
    supervised_indices = []
    indices_to_expert = []
    bg_indices = []

    #Next new batch size:
    myB = np.shape(X_new)[0]

    #Calculate n_top and n_active, the number of items in each loop that will be sent to
    #the expert for anomaly prediction and active learning, respectively.
    n_top = int(np.ceil(pc_top*n_send))
    n_active = n_send - n_top

    #print('Given your choice of parameters, the supervised classifier will send',n_top,'and the active learner will send',n_active,'candidates to the expert for labeling from each batch.')

    #Concatenate the old data to the new data:
    X_curr = np.concatenate([X_old,X_new],axis=0)

    # Send at most the last n_data_max rows to the models for scoring (in order to keep
    # run time reasonable):
    if np.shape(X_curr)[0] > n_data_max:
        X_curr = X_curr[-n_data_max:,:]


    #######################################################################################
    # CALCULATING SCORES ##################################################################
    #######################################################################################

    # We require there to be at least n_data_min data points before calculating scores
    if np.shape(X_curr)[0] >= n_data_min:

        # Define array to stock all current scores. Note that if X_curr contains data from the
        # original X_old, their scores will be calculated ANEW here BUT if there had been sufficient
        # labeled data in X_old originally to start to fill in "all_labeled_scores" previously,
        # WE DO NOT UPDATE "all_labeled_scores" based on the recalculation of scores here for
        # the old data points, only for the new batch's data points.

        # Calculate scores for all current data and all models:
        all_scores = np.empty([np.shape(X_curr)[0],len(models.items())])

        # We run through the unsupervised models, one by one, to output scores:
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_curr)
            y_score = model.score_samples(X_curr)
            y_score.dtype = np.float64
            all_scores[:,i] = y_score.squeeze()

        # For simplicity, we only keep the scores corresponding to new data:
        all_scores = all_scores[-myB:,:]

        # Supervised learning interlude:

        # If we have sufficient labeled data
        # (either from the initialization step or from PREVIOUS loops of the
        # present function), AND we have the required number of labeled item from each class,
        # we run the machine learning method chosen.

        if len(Y_lab) >= min_n_labeled and sum([1 for i in range(len(Y_lab)) if Y_lab[i]==0]) >= min_n_nom and sum([1 for i in range(len(Y_lab)) if Y_lab[i]==1]) >= min_n_anom:

            # Create and train the supervised model
            model = classifier_class(**classifier_params)
            learned_model = model.fit(all_labeled_scores, Y_lab)

            # Get predictions
            new_preds = learned_model.predict_proba(all_scores)[:, 1]

            # We have to then sort "new_preds" and get the indices of the n_send largest:
            sorted_indices = np.argsort(new_preds)
            if n_top > 0:
                supervised_indices = sorted_indices[-n_top:].tolist()
            else:
                supervised_indices = []

        else:
            learned_model = None


        #######################################################################################
        # ACTIVE LEARNING #####################################################################
        #######################################################################################

        # Here we implement a module that performs active learning. There are an enormous
        # number of ways to go about this, whether it be which strategy to use, but also from
        # which pool of data do we look for candidates.

        # We shall do active learning by wrapping around the modAL Python package, which
        # implements active learning by itself typically wrapping around methods in scikit-learn.

        # For the moment, we use only the data in
        # X_new as potential data to give to the expert, given the labeled data we already have
        # from previous loops (if any). In the next phase of code development we might
        # also include ALL previous unlabeled data in this section.

        # First we need to know if there is already some labeled data or not. If there is
        # so far NO labeled data, or only data labeled from one class, we propose to use
        # "random sampling" as active learning.

        active_indices = []
        curr_all_scores = all_scores

        if len(Y_lab) > min_n_labeled and sum([1 for i in range(len(Y_lab)) if Y_lab[i]==0]) >= min_n_nom and sum([1 for i in range(len(Y_lab)) if Y_lab[i]==1]) >= min_n_anom:

            curr_all_labeled_scores = all_labeled_scores
            curr_Y_lab = Y_lab

            # Set up the chosen active learning strategy (default is margin_sampling)
            # Create classifier instance with the provided parameters
            classifier_instance = classifier_class(**classifier_params)

            # Set up the ActiveLearner
            learner = ActiveLearner(
                estimator=classifier_instance,
                query_strategy=margin_sampling,
                X_training=curr_all_labeled_scores,
                y_training=curr_Y_lab
            )

            #Since we run active learning using the modAL package, unfortunately it only
            #outputs one candidate at a time, so there is a whole lot of work to do to make sure
            #nothing stupid happens with indices as we iteratively output one candidate at a time
            #until we get to n_active:
            true_indices = list(range(0,np.shape(curr_all_scores)[0]))

            for i in range(n_active):

                query_idx, query_inst = learner.query(curr_all_scores)
                query_idx = int(query_idx[0])

                # Append new index to list:
                active_indices.append(true_indices[query_idx])

                #Remove this index from true_indices:
                del true_indices[query_idx]

                # Remove this guy from the data and loop back:
                curr_all_scores = np.delete(curr_all_scores, query_idx, axis=0)

        else:
            # There is no (or not enough) labeled data, so we do random sampling:
            my_random_permute = np.random.permutation(np.shape(curr_all_scores)[0])
            active_indices = [int(my_random_permute[i]) for i in range(n_active)]


        #Note that it is possible that there is an overlap in the sets supervised_indices and
        #active_indices, which will mean that indices to expert may not contain n_send elements
        #in total; it may contain less.
        #Combine indices from supervised, active learning
        indices_to_expert = supervised_indices + active_indices
        indices_to_expert = sorted(set(indices_to_expert))

        #######################################################################################
        # UPDATING PARAMETERS #################################################################
        #######################################################################################

        # Update all_labeled_scores:
        if np.shape(all_labeled_scores) != (0,0):
            all_labeled_scores = np.concatenate([all_labeled_scores,all_scores[indices_to_expert,:]],axis=0)

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

    return X_old, X_lab, all_labeled_scores, indices_to_expert, learned_model, supervised_indices, active_indices

def aaa():
    #TODO: unique point d'entrée ?
    return 42

def Compute_All_Model_Scores(X, models):
    """
    Compute anomaly scores for each model in the models dictionary.
    Parameters:
    - X: The data to compute the scores for (shape: [n_samples, n_features]).
    - models: Dictionary of models to compute scores from.
    Returns:
    - scores_array: A NumPy array where each row corresponds to a data point
    and each column corresponds to a model's score for that data point.
    """

    # Initialize the scores array with the correct shape: n_samples x n_models
    n_samples = X.shape[0]
    n_models = len(models)
    scores_array = np.zeros((n_samples, n_models))
    # Loop over each model and calculate the scores
    for i, (model_name, model) in enumerate(models.items()):
        model.fit(X)
        scores = model.score_samples(X).ravel() # Flatten the scores to a 1D array
        scores_array[:, i] = scores
    return scores_array
