import numpy as np
import acanag.sampling as samp
import acanag.loda_utils as loda
import acanag.acanag as aaa
from sklearn.ensemble import RandomForestClassifier
from modAL.uncertainty import margin_sampling

# The number of "old" data points we start with:
n_old = 1000

# The number of data points in each future batch:
B = 500

# The number of future batches:
n_loops = 50

# The mixture parameter:
tau = 0.01

#############################################################################
# n-dimensional Gaussian nominals with one ten-dimensional Gaussian anomaly 
#############################################################################

#Set n:
num_dim = 10

def scaled_identity_matrix(num_dim,c_m):
    return [[c_m if i == j else 0 for j in range(num_dim)] for i in range(num_dim)]

# Specific arguments:
a_list = [[1]*num_dim]

anomaly_cov_list = [scaled_identity_matrix(num_dim,1)]

nominal_mean = np.array([0]*num_dim)    # Mean of the nominal Gaussian distribution
nominal_cov = np.array(scaled_identity_matrix(num_dim,1))   # Covariance of the nominal Gaussian distribution

# Sampling
X, Y = samp.sample_data(
    samp.multivariate_gaussian_sampling_with_anomaly_gaussians, # Sampling scheme
    n_old=n_old,                                           # Initial number of data points
    B=B,                                                   # Batch size
    n_loops=n_loops,                                       # Number of batches
    tau=tau,                                               # Fraction of anomalies
    a_list=a_list,                                         # Anomaly means
    anomaly_cov_list=anomaly_cov_list,                     # Anomaly covariance matrices
    nominal_mean=nominal_mean,                             # Nominal mean
    nominal_cov=nominal_cov                               # Nominal covariance matrix
)

#Proportion of initial data we suppose we know the true anomaly status of:
u = 0.1

n_old_anomalies = np.sum(Y[:n_old] == 1)
n_old_nominals = n_old - n_old_anomalies
#Calculate how many labels to show and how many to hide:
n_hide = int(np.ceil((1-u)*n_old)) # if u > 0 then at least one label will be shown due to ceiling function
n_hide_pc = (n_hide/n_old)*100
#Randomly select n_hide of the n_old data points to mask:
permute_indices = np.random.permutation(n_old)
# The n_anomalies indices in permute_indices will correspond to the anomalies:
hide_indices = permute_indices[0:n_hide]
#Fill in Y_muted:
Y_muted = Y[:n_old].astype(float)
Y_muted[hide_indices] = np.nan
n_remaining_anomalies = np.sum(Y_muted == 1)
n_remaining_nominals = np.sum(Y_muted == 0)
print('There were',n_old_anomalies,'anomalies and',n_old_nominals,'nominals in the initial data. After randomly masking',n_hide_pc,'% of the initial data, there remain',n_remaining_anomalies,'labeled anomalies and',n_remaining_nominals,'labeled nominals.')

#Minimum number of data points to use to calibrate the number of LODA projections, unless the 
#today number of data points is less than n_min. The choice between the two happens elsewhere
#so this can be left here.
n_min = 2000

#Set an upper bound for the number of LODA projections:
M_max = 10 #500

#Parameter set to default 0.01 in Pevny (2015). However, in small-dimensional settings (e.g., d=2) this
#may not be a good idea?
tau_M = 0.01

#We also need to decide how many data-points to feed to the expert in each loop IN TOTAL. In
#Das et al. (2016) they feed the top one. Here there are different algorithms, and not all of the
#have an active learning step. What we do is define the total number of items the expert can see in
#each pass = n_send. However, for instance with our method, not all of these will be sent because
#they had high scores (= possible anomaly). Some of them will be kept aside to be used in an active
#learning strategy (e.g., uncertainty sampling)
n_send = 5

#Simply use the first n_min data-points in X to do this.
models, best_m, scores = loda.LODA_Choose_M(X[:min(n_min,n_old+B*n_loops),:],M_max=M_max,tau_M=tau_M)

#############

#attempt:
X_old, X_lab, Y_lab, all_labeled_scores = aaa.init_super_sad(X[:n_old,:],Y[:n_old].tolist(),data_dim = num_dim,n_data_min = 100, models=models)
aaa.super_sad(X_new = X[n_old:(n_old+100),:], X_old = X_old, X_lab = X_lab, Y_lab = Y_lab, all_labeled_scores = all_labeled_scores, models=models, supervised_model={'class': RandomForestClassifier, 'params': {'class_weight': {0: 0.999, 1: 0.001}}}, query_strategy = margin_sampling, n_data_min = 100, n_data_max = 2000, min_n_labeled = 5, n_send=2, pc_top = 0.5, min_n_nom=5, min_n_anom=1)
