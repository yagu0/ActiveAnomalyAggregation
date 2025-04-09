import matplotlib.pyplot as plt

def plot_density_estimations(raw_scores, Y_raw_scores, tau=0.01):
    
    """
    This function takes an array raw_scores of raw scores from
    a set of M anomaly models, along with a 1d array Y_raw_scores of labels for them
    (0 for nominal and 1 for anomaly) and plots a kind of mixture density plot for 
    each anomaly detector, where the mixture coefficient (estimated, known, or guessed) 
    is called tau.

    Parameters:
    - raw_scores: array of raw scores from a set of M anomaly detectors of size
        (number of data points concerned) x (number of anomaly detectors M)
    - Y_raw_scores: known labels (0 or 1) corresponding to the data points that are
        associated with the scores in raw_scores
    - tau: mixture coefficient, either estimated, known, or guessed. You can artificially
        inflate it for visual purposes if you cannot see the anomaly density when tau
        is too small.

    Returns: a plot of subplots, one for each anomaly detector
    """
    
    M = raw_scores.shape[1]  # Number of models (columns)
    
    # Determine subplot grid size
    if M <= 3:
        rows, cols = 1, M
    elif M <= 6:
        rows, cols = 2, (M + 1) // 2
    elif M <= 10:
        rows, cols = 3, (M + 2) // 3
    else:
        rows, cols = 4, (M + 3) // 4  # For M > 10, use 4 rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten() if M > 1 else [axes]  # Flatten axes for easy indexing

    for i in range(M):
        ax = axes[i]  # Handle multiple subplots

        # Extract scores for the current model
        scores = raw_scores[:, i]

        # Split into nominal (Y_AUC = 0) and anomaly (Y_AUC = 1)
        nominal_scores = scores[Y_raw_scores == 0]
        anomaly_scores = scores[Y_raw_scores == 1]

        # Generate KDE with proper scaling
        x_min, x_max = np.min(scores), np.max(scores)
        x_vals = np.linspace(x_min, x_max, 200)  # Common x-axis range

        if len(nominal_scores) > 1:  # Ensure enough points
            kde_nominal = gaussian_kde(nominal_scores)
            nominal_density = kde_nominal(x_vals) * (1 - tau)  # Scale by (1 - tau)
            ax.plot(x_vals, nominal_density, label=f"Nominal (Y=0)", color="blue")

        if len(anomaly_scores) > 1:  # Ensure enough points
            kde_anomaly = gaussian_kde(anomaly_scores)
            anomaly_density = kde_anomaly(x_vals) * tau  # Scale by tau
            ax.plot(x_vals, anomaly_density, label=f"Anomaly (Y=1)", color="red")

        # Formatting
        ax.set_title(f"Density Estimation - Model {i+1}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Density (Scaled)")
        ax.legend()
        ax.grid(True)

        # Set y-axis limits
        ax.set_ylim(0, 0.1)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
