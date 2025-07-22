import numpy as np
import math
import random

def optimal_bin_count(my_projections):
    """
    Finds the optimal number of bins for a histogram of `my_projections` by maximizing a criterion.

    Parameters:
    - my_projections: numpy array of shape (N, 1), the real numbers to be binned.

    Returns:
    - optimal_b: The value of b (number of bins) that maximizes the criterion.
    """
    N = len(my_projections)
    my_projections = my_projections.flatten()  # Ensure it's a 1D array
    min_val, max_val = np.min(my_projections), np.max(my_projections)
    best_b = 1
    max_criterion = -np.inf  # Initialize to a very low value
    
    # Try bin counts from 1 to N
    for b in range(1, N + 1):
        # Generate bin edges from min_val to max_val with b bins
        bin_edges = np.linspace(min_val, max_val, b + 1)
        # Compute histogram counts (n_i), ensuring edges include min and max
        counts, _ = np.histogram(my_projections, bins=bin_edges)
        
        # Calculate the criterion
        criterion = 0
        for n_i in counts:
            if n_i > 0:  # Avoid log(0)
                criterion += n_i * np.log(b * n_i / N)
        
        # Subtract the penalty term
        criterion -= (b - 1 + np.log(b) ** 2.5)
        #print(criterion)
        
        # Update the best value
        if criterion > max_criterion:
            max_criterion = criterion
            best_b = b
    
    return best_b

def process_histogram_and_assign_values(my_projections, best_b):
    """
    Constructs a histogram with best_b bins, calculates empirical densities,
    and assigns -log(empirical density) to each value in my_projections.

    Parameters:
    - my_projections: numpy array of shape (N, 1), the projections.
    - best_b: int, the optimal number of bins for the histogram.

    Returns:
    - result_array: numpy array of shape (N, 1), values corresponding to
      -log(empirical density) for each element in my_projections.
    """
    # Flatten my_projections to ensure it's 1D for histogram operations
    my_projections = my_projections.flatten()

    # Get min and max values for bin boundaries
    min_val = np.min(my_projections)
    max_val = np.max(my_projections)
    
    # Construct histogram
    bin_edges = np.linspace(min_val, max_val, best_b + 1)
    counts, _ = np.histogram(my_projections, bins=bin_edges)
    
    # Calculate empirical densities
    empirical_probas = counts / len(my_projections)
    
    # Assign -log(empirical density) to each value in my_projections
    result_array = np.zeros_like(my_projections)
    for i, val in enumerate(my_projections):
        # Find the bin index for each value
        bin_index = np.searchsorted(bin_edges, val, side='right') - 1
        bin_index = min(bin_index, best_b - 1)  # Handle edge case for max_val
        
        # Assign -log(empirical density)
        result_array[i] = -np.log(empirical_probas[bin_index])
    
    return result_array.reshape(-1, 1)  # Reshape to (N, 1)

class LODA:
    def __init__(self, proj_vect):
        # proj_vect must be of shape (n dim)x(1)
        self.proj_vect = proj_vect
    
    def fit(self, X):
        # Just fit on X without modifying proj_vect
        self.X = X
        return self
    
    def score_samples(self, X):
        
        #Project all of the data to 1d using the projection vector:
        my_projections = np.matmul(self.X,self.proj_vect)
        #print(my_projections)
        
        
        #Construct a 1d histogram with respect to this data:
        #1) Get min and max data points:
        x_min = np.min(my_projections)
        x_max = np.max(my_projections)
        
        
        best_b = optimal_bin_count(my_projections)
        #print('The best b is: ',best_b)
        #plt.hist(my_projections,bins=best_b)
        
        scores = process_histogram_and_assign_values(my_projections, best_b)
        
        return scores

def add_model_variant(models, all_proj_vectors, i1, i2):
    """
    Adds a model to the models dictionary with dynamic key and index.
    
    Parameters:
    - models: Dictionary to store models.
    - all_proj_vectors: 2D numpy array with projection vectors.
    - i: Integer index for accessing the projection vector and naming the model.
    """
    
    models[f"Loda_{i1}"] = LODA(proj_vect=all_proj_vectors)

def LODA_OAT(M=None,d=None,models=None,LODA_index=None):
    
    # M is the number of LODA models we want
    # d is the dimension of the data
    # models is a dictionary (perhaps empty) of models
    # LODA_index is the index we want the LODA model to have (e.g., LODA_3)
    
    
    #If there are no models so far:
    if models==None:
        
        #Define an empty dictionary:
        models = {}
    
    #Initialize array to store all the projection vectors:
    all_proj_vectors = np.zeros((d,M))
    
    #Create LODA models one at a time and add them to the dictionary:
    for i in range(M):
        
        #Generate a weights vector with N(0,1) entries:
        w = np.random.normal(loc=0,scale=1,size = (d,1))
        
        #Keep only ceil(sqrt(d)) of them
        d_keep = math.ceil(np.sqrt(d))
        d_discard = d - d_keep
        
        #Randomly choose d_discard out of d entries and set them to 0:
        which_discard = random.sample(range(0,d),d_discard)
        for j in which_discard:
            w[j] = 0
        
        all_proj_vectors[:,i] = w.squeeze()
        #Add this model to the dictionary:
        add_model_variant(models, all_proj_vectors[:,i], LODA_index,i)
        
    #print(all_proj_vectors)
        
        
    return models, all_proj_vectors

def LODA_Choose_M(X,M_max=50,tau_M=0.01):
    
    #############################################################################################
    # X: the original data we are given to work with. (rows= N data-points, cols = dimensions)
    # M_max: some upper limit on the maximum number of anomaly detectors we are
    #        willing to consider (default = 50)
    # tau_M: a calibration parameter (default in Pevny (2016) is 0.01) unrelated to tau elsewhere
    #############################################################################################
    
    # Initialize the dictionary called models:
    models = {}
    
    # Initialize a numpy array to stock all the scores from the M_max potential ensemble members:
    unweighted_scores = np.zeros((np.shape(X)[0],M_max))

    # We first need to get the first and second function calls so that we can
    # calculate sigma_1:
    next_model_1, proj_vect = LODA_OAT(M=1,d=np.shape(X)[1],models=None,LODA_index=1)
    print('First projection vector:',proj_vect)
    for i, (name, model) in enumerate(next_model_1.items()):
        model.fit(X)
        y_score = model.score_samples(X)
        y_score.dtype = np.float64
        unweighted_scores[:,0] = y_score.squeeze()
        
    next_model_2, proj_vect = LODA_OAT(M=1,d=np.shape(X)[1],models=None,LODA_index=2)
    print('Second projection vector:',proj_vect)
    for i, (name, model) in enumerate(next_model_2.items()):
        model.fit(X)
        y_score = model.score_samples(X)
        y_score.dtype = np.float64
        unweighted_scores[:,1] = y_score.squeeze()
        
    # Add these two models to the dictionary called models:
    models.update(next_model_1)
    models.update(next_model_2)

    # Calculate sigma_1:
    sigma_1 = (1/np.shape(X)[0])*np.sum( abs(  (1/2)*unweighted_scores[:,0] + (1/2)*unweighted_scores[:,1] - unweighted_scores[:,0] )   )
    
    # Now we shall continue, adding one column at a time to unweighted scores until the condition
    # in Pevny (2015) is satisfied:

    m = 2
    sigma_frac = 1e10
    while m < M_max and sigma_frac > tau_M:
        
        m = m+1
        
        next_model, all_proj_vectors = LODA_OAT(M=1,d=np.shape(X)[1],models=None,LODA_index=m)
        print(m,'th projection vector:',all_proj_vectors)
        
        for i, (name, model) in enumerate(next_model.items()):
            model.fit(X)
            y_score = model.score_samples(X)
            y_score.dtype = np.float64
            unweighted_scores[:,(m-1)] = y_score.squeeze()
            
        # Add this model to the dictionary called models:
        models.update(next_model)
        
        # Calculate sigma_(m-1):
        sigma_new = (1/np.shape(X)[0]) * np.sum( abs( np.sum((1/m)*unweighted_scores[:,0:m],axis=1) - np.sum((1/(m-1))*unweighted_scores[:,0:(m-1)],axis=1)     ) )
        
        # Calculate sigma_(m-1)/sigma_1:
        sigma_frac = sigma_new/sigma_1
        print(sigma_frac)
        
    return models, m, unweighted_scores[:,0:(m-1)]
