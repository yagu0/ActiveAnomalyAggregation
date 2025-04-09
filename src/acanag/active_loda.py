import numpy as np

def optimize_w(H_A, H_N, q_tau, C_A=100, C_eta=1000):
    
    """
    What we call "active-LODA" is the method described in the article below:

    Shubhomoy Das, Weng-Keen Wong, Thomas Dietterich, Alan Fern, and Andrew
    Emmott. Incorporating expert feedback into active anomaly discovery. In 2016 IEEE
    16th International Conference on Data Mining (ICDM), pages 853–858. IEEE, 2016.

    Essentially, active-LODA optimizes the weight vector w that gives the linear
    coefficient values to reweigh the raw scores
    H_A of currently-known anomalies and H_N of currently-known nominals. This function
    is called in each loop when a new batch of data arrives.

    Remark: this algorithm may struggle to optimize in various, hard to predict, situations.
    For instance, it starts to struggle when the size of H_A and H_N get too large (a few
    hundred rows) and may start to output warnings about convergence. 

    Parameters:
    - H_A: A numpy array of shape (n_A, M) representing the current set of 
        unweighted scores of known anomalies.
    - H_N: A numpy array of shape (n_N, M) representing the current set of
        unweighted scores of known nominals.
    - q_tau: the weighted sum score, after ordering from smallest to largest in
        the current batch, correspondingbto the (1-C_tau)-th quantile of the 
        set of weighted sum scores.
    - C_A: constant, suggested = 100 in Das et al. (2016)
    - C_eta: constant, suggested = 1000 in Das et al. (2016)

    Returns:
    - w: The optimal weights vector of length M.
    """
    
    M = H_A.shape[1]  # Length of each z_i (dimension of w)
    n_A = H_A.shape[0]  # Number of elements in H_A
    n_N = H_N.shape[0]  # Number of elements in H_N
    
    c1 = C_A/n_A
    c2 = 1/n_N
    c3 = C_eta
    
    # Initialize w_M: Vector where each entry is 1/sqrt(M)
    w_M = np.ones(M) / np.sqrt(M)
    
    # Decision variable for w
    w = cp.Variable(M)

    # Slack variables eta_{ij} for each pair (z_i, z_j)
    eta = cp.Variable((n_A, n_N), nonneg=True)

    # Define the loss function L
    def loss(z, y, w, q_tau):
        wTz = w @ z  # Inner product w^T * z
        return cp.max(cp.hstack([
            cp.multiply(y, cp.pos(q_tau - wTz)),  # Loss for y_i = 1
            cp.multiply(1 - y, cp.pos(wTz - q_tau))  # Loss for y_i = 0
        ]))

    # Compute losses for H_A and H_N
    loss_H_A = cp.sum([loss(z, 1, w, q_tau) for z in H_A])
    loss_H_N = cp.sum([loss(z, 0, w, q_tau) for z in H_N])

    # New soft constraints for eta_{ij}
    soft_constraints = []
    for i, z_i in enumerate(H_A):
        for j, z_j in enumerate(H_N):
            # Soft constraint: w^T * (z_i - z_j) + eta_{ij} >= 0
            soft_constraints.append(w @ (z_i - z_j) + eta[i, j] >= 0)
    
    objective = (
        c1 * loss_H_A
        + c2 * loss_H_N
        + c3 * cp.sum(eta)
        + (1/10)*c3*M*((1+n_A)*(1+n_N))*cp.sum_squares(w - w_M)  # L2 regularization
    )

    # Problem definition
    problem = cp.Problem(cp.Minimize(objective), soft_constraints)

    # Solve the problem
    result = problem.solve()

    # Output results: Return the optimal w
    return w.value
