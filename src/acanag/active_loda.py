import numpy as np
import cvxpy as cp

def optimize_w_2(H_A, H_N, q_tau, C_A=100, C_eta=1000):
    
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
    
    M = H_A.shape[1]
    n_A = H_A.shape[0]
    n_N = H_N.shape[0]

    c1 = C_A / n_A
    c2 = 1 / n_N
    c3 = C_eta

    w_M = np.ones(M) / np.sqrt(M)

    # Decision variable for w
    w = cp.Variable(M)

    # Slack variables eta_{ij} for each pair (z_i, z_j)
    eta = cp.Variable((n_A, n_N), nonneg=True)

    # === Vectorized Loss Computation ===
    scores_H_A = H_A @ w            # shape (n_A,)
    scores_H_N = H_N @ w            # shape (n_N,)

    loss_H_A = cp.sum(cp.pos(q_tau - scores_H_A))  # loss when y = 1
    loss_H_N = cp.sum(cp.pos(scores_H_N - q_tau))  # loss when y = 0

    # === Vectorized Soft Constraints ===
    z_diff = H_A[:, np.newaxis, :] - H_N[np.newaxis, :, :]     # shape (n_A, n_N, M)
    z_diff_flat = z_diff.reshape(-1, M)                        # shape (n_A*n_N, M)
    eta_flat = cp.reshape(eta, (n_A * n_N,))                   # shape (n_A*n_N,)

    soft_constraints = [z_diff_flat @ w + eta_flat >= 0]

    # === Objective Function ===
    objective = (
        c1 * loss_H_A
        + c2 * loss_H_N
        + c3 * cp.sum(eta)
        +  c3 * (1 + n_A) * (1 + n_N) * cp.sum_squares(w - w_M)
    )

    # Problem definition and solve
    problem = cp.Problem(cp.Minimize(objective), soft_constraints)
    result = problem.solve()

    return w.value
