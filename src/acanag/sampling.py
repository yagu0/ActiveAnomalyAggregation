import numpy as np

#example:
# X, Y = sample_data.sample_data(sample_data.multivariate_gaussian_sampling, a_list=([1,0,0],[0,1,0]), nominal_mean=np.array([0,0,0]), nominal_cov=np.identity(3))

def sample_data(sampling_scheme, n_old=1000, B=1000, n_loops=100, tau=.01, **kwargs):
    """
    Sample data using the chosen sampling scheme.

    Parameters:
    - sampling_scheme (callable): The function or method that implements the sampling logic.
    - n_old: number of data-points we are give to start with
    - B: batch size for new data
    - n_loops: total number of batches
    - tau: value between 0 and 1 corresponding to the % of the data that is an anomaly
           (usually close to 1)
    - **kwargs: Additional parameters specific to the chosen sampling scheme.

    Returns:
    - np.ndarray: The sampled data.
    """
    if not callable(sampling_scheme):
        raise ValueError("The sampling_scheme must be a callable (function or class).")

    # quantity of data required to simulate:
    n_data = n_old + B*n_loops

    # Call the sampling scheme with the provided size and other arguments
    return sampling_scheme(n_data,tau, **kwargs)


def uniform_sampling_point_mass_n_dim(n_data, tau, a_list, lower=0, upper=1):
    """
    Generate nominal samples uniformly in a given range, with anomalies split between multiple fixed values.

    Parameters:
    - n_data: Number of samples.
    - tau: Fraction of the data which is an anomaly (0 <= tau <= 1).
    - a_list: List of lists, where each list contains an L-dimensional anomaly value.
    - lower: Lower bound of the range.
    - upper: Upper bound of the range.

    Returns:
    - X: np.ndarray of sampled points.
    - Y: np.ndarray of corresponding labels (1 for anomalies, 0 for nominal points).
    """

    # Ensure that we have more than 0 constants
    n = len(a_list)
    if n == 0:
        raise ValueError("a_list must contain at least one value.")

    # Set L from a_list[0]
    L = len(a_list[0])

    # Convert a_list (list of lists) into a list of numpy arrays inside the function
    a_list = [np.array(a) if not isinstance(a, np.ndarray) else a for a in a_list]

    # Validate that all values in a_list are between 'lower' and 'upper'
    if any(np.any(a < lower) or np.any(a > upper) for a in a_list):
        raise ValueError("All values in each array of a_list must be between 'lower' and 'upper'.")

    # Determine number of anomalies
    n_anomalies = np.random.binomial(n_data, tau)
    if n_anomalies == 0:
        raise ValueError("There are no true anomalies. Try with a larger tau or n_data.")

    # Randomly permute indices and split into anomalies and nominals
    permute_indices = np.random.permutation(n_data)
    anomaly_indices = permute_indices[:n_anomalies]
    nominal_indices = permute_indices[n_anomalies:]

    # Initialize the data array
    X = np.empty((n_data, L))

    # Determine how many anomalies each constant should handle
    n_per_constant = n_anomalies // n
    remainder = n_anomalies % n

    # Split anomalies between the constants
    start_idx = 0
    for i in range(n):
        # For the last constant, add the remainder anomalies
        n_for_this_constant = n_per_constant + (1 if i < remainder else 0)

        # Assign this portion of anomalies to the current constant (multidimensional)
        X[anomaly_indices[start_idx:start_idx + n_for_this_constant]] = a_list[i]
        start_idx += n_for_this_constant

    # Assign nominal points with random uniform values
    X[nominal_indices] = np.random.uniform(lower, upper, size=(len(nominal_indices), L))

    # Create labels for anomalies and nominals
    Y = np.empty(n_data, dtype=int)
    Y[anomaly_indices] = 1
    Y[nominal_indices] = 0

    return X, Y

########################################################################################

def multivariate_gaussian_sampling(n_data, tau, a_list, nominal_mean, nominal_cov):
    """
    Generate nominal samples using a multivariate Gaussian distribution for nominals,
    with anomalies split between multiple fixed constant values.

    Parameters:
    - n_data: Number of samples.
    - tau: Fraction of the data which is an anomaly (0 <= tau <= 1).
    - a_list: List of L-dimensional constant anomaly values (one for each anomaly type).
    - nominal_mean: L-dimensional mean vector for the Gaussian distribution of nominal data.
    - nominal_cov: LxL covariance matrix for the Gaussian distribution of nominal data.

    Returns:
    - X: np.ndarray of sampled points (size: n_data x L).
    - Y: np.ndarray of corresponding labels (1 for anomalies, 0 for nominal points).
    """

    # Ensure that we have more than 0 constants in a_list
    n = len(a_list)
    if n == 0:
        raise ValueError("a_list must contain at least one value.")

    # Set L from a_list[0]
    L = len(a_list[0])

    # Convert a_list (list of lists) into a list of numpy arrays inside the function
    a_list = [np.array(a) if not isinstance(a, np.ndarray) else a for a in a_list]

    # Validate that nominal_mean and nominal_cov are correct shapes
    if nominal_mean.shape != (L,):
        raise ValueError(f"nominal_mean must be a vector of length {L}.")
    if nominal_cov.shape != (L, L):
        raise ValueError(f"nominal_cov must be a {L}x{L} matrix.")

    # Ensure covariance matrix is positive semi-definite (for Gaussian distributions)
    if not np.all(np.linalg.eigvals(nominal_cov) >= 0):
        raise ValueError("nominal_cov must be positive semi-definite.")

    # Determine the number of anomalies
    n_anomalies = np.random.binomial(n_data, tau)
    if n_anomalies == 0:
        raise ValueError("There are no true anomalies. Try with a larger tau or n_data.")

    # Randomly permute indices and split into anomalies and nominals
    permute_indices = np.random.permutation(n_data)
    anomaly_indices = permute_indices[:n_anomalies]
    nominal_indices = permute_indices[n_anomalies:]

    # Initialize the data array (L dimensions)
    X = np.empty((n_data, L))

    # Determine how many anomalies each constant should handle
    n_per_constant = n_anomalies // n
    remainder = n_anomalies % n

    # Split anomalies between the constants and assign values
    start_idx = 0
    for i in range(n):
        # For the first constants, add the remainder anomalies
        n_for_this_constant = n_per_constant + (1 if i < remainder else 0)

        # Assign the anomaly values directly from a_list (multidimensional)
        X[anomaly_indices[start_idx:(start_idx + n_for_this_constant)]] = a_list[i]
        start_idx += n_for_this_constant

    # Assign nominal points by sampling from the multivariate Gaussian distribution
    X[nominal_indices] = np.random.multivariate_normal(nominal_mean, nominal_cov, len(nominal_indices))

    # Create labels for anomalies and nominals
    Y = np.empty(n_data, dtype=int)
    Y[anomaly_indices] = 1
    Y[nominal_indices] = 0

    return X, Y

########################################################################################

def uniform_sampling_point_mass_with_epsilon_n_dim(n_data, tau, a_list, epsilon, lower=0, upper=1):
    """
    Generate nominal samples from a uniform distribution, with anomalies distributed
    around a set of centers defined by a_list. Anomalies are distributed in uniform
    volumes (L-dimensional cubes) around each center with the size of the volume determined by epsilon.

    Parameters:
    - n_data: Number of samples.
    - tau: Fraction of the data which is an anomaly (0 <= tau <= 1).
    - a_list: List of L-dimensional centers for anomalies. Length of a_list defines the number of different anomaly volumes.
    - epsilon: The size of the volume around each center, determining the width of the anomaly distribution in each dimension.
    - lower: Lower bound of the range for nominal data (for validation).
    - upper: Upper bound of the range for nominal data (for validation).

    Returns:
    - X: np.ndarray of sampled points (n_data x L).
    - Y: np.ndarray of corresponding labels (1 for anomalies, 0 for nominal points).
    """

    # Ensure a_list is not empty
    if len(a_list) == 0:
        raise ValueError("a_list must contain at least one value.")

    # Set L from a_list[0]
    L = len(a_list[0])

    # Ensure that epsilon is a positive value
    if epsilon <= 0:
        raise ValueError("epsilon must be a positive value.")

    # Validate that the anomaly ranges do not exceed the bounds
    for center in a_list:
        if len(center) != L:
            raise ValueError(f"Each center in a_list must be a {L}-dimensional point.")

        # Ensure the box around the center does not exceed the bounds [lower, upper]
        if not all(lower <= center[i] - epsilon and center[i] + epsilon <= upper for i in range(L)):
            raise ValueError(f"Center {center} with epsilon {epsilon} creates a range outside of "
                             f"the bounds [{lower}, {upper}] for dimension {i}.")

    # Ensure that the volumes around the centers do not overlap
    for i in range(len(a_list)):
        for j in range(i + 1, len(a_list)):
            center_1 = np.array(a_list[i])
            center_2 = np.array(a_list[j])

            # Check overlap: The volumes will overlap if the distance in every dimension is less than 2 * epsilon
            overlap = all(
                abs(center_1[dim] - center_2[dim]) < 2 * epsilon
                for dim in range(L)
            )

            if overlap:
                raise ValueError(f"The volumes around centers {a_list[i]} and {a_list[j]} overlap. "
                                 f"Ensure that the centers are at least {2 * epsilon} apart in at least one dimension.")

    # Calculate the number of anomalies
    n_anomalies = np.random.binomial(n_data, tau)
    if n_anomalies == 0:
        raise ValueError("There are no true anomalies. Try with a larger tau or n_data.")

    # Randomly permute indices and split into anomalies and nominals
    permute_indices = np.random.permutation(n_data)
    anomaly_indices = permute_indices[:n_anomalies]
    nominal_indices = permute_indices[n_anomalies:]

    # Initialize the data array for X (with L-dimensional samples)
    X = np.empty((n_data, L))

    # Determine how many anomalies to assign to each center in a_list
    n_centers = len(a_list)
    n_per_center = n_anomalies // n_centers
    remainder = n_anomalies % n_centers

    # Handle any remainders in anomaly assignment
    anomalies_per_center = np.full(n_centers, n_per_center)
    for i in range(remainder):
        anomalies_per_center[i] += 1

    # Assign anomalies to corresponding centers in a_list
    start_idx = 0
    for i, center in enumerate(a_list):
        # For each center, generate the anomaly points by sampling uniformly within the volume
        # Create a random offset for each dimension within the range [-epsilon, epsilon]
        lower_range = np.array(center) - epsilon
        upper_range = np.array(center) + epsilon

        # Sample the anomalies uniformly within the volume around each center
        X[anomaly_indices[start_idx:start_idx + anomalies_per_center[i]], :] = \
            np.random.uniform(lower_range, upper_range, size=(anomalies_per_center[i], L))

        start_idx += anomalies_per_center[i]

    # Assign nominal points with random uniform values in the range [lower, upper]
    X[nominal_indices] = np.random.uniform(lower, upper, size=(len(nominal_indices), L))

    # Create labels for anomalies (1) and nominals (0)
    Y = np.empty(n_data, dtype=int)
    Y[anomaly_indices] = 1
    Y[nominal_indices] = 0

    return X, Y

########################################################################################

def multivariate_gaussian_sampling_with_epsilon_balls(n_data, tau, a_list, epsilon, nominal_mean, nominal_cov):
    """
    Generate nominal samples using a multivariate Gaussian distribution for nominals,
    with anomalies sampled uniformly within L-dimensional balls of radius epsilon
    around given centers.

    Parameters:
    - n_data: Number of samples.
    - tau: Fraction of the data which is an anomaly (0 <= tau <= 1).
    - a_list: List of L-dimensional centers for anomalies.
    - epsilon: Radius of the L-dimensional balls around each center for anomalies.
    - nominal_mean: L-dimensional mean vector for the Gaussian distribution of nominal data.
    - nominal_cov: LxL covariance matrix for the Gaussian distribution of nominal data.

    Returns:
    - X: np.ndarray of sampled points (size: n_data x L).
    - Y: np.ndarray of corresponding labels (1 for anomalies, 0 for nominal points).
    """

    # Ensure a_list is not empty
    if len(a_list) == 0:
        raise ValueError("a_list must contain at least one value.")

    # Set L from a_list[0]
    L = len(a_list[0])

    # Validate that a_list contains the correct number of dimensions
    for center in a_list:
        if len(center) != L:
            raise ValueError(f"Each element in a_list must be {L}-dimensional.")

    # Convert a_list (list of lists) into a list of numpy arrays inside the function
    a_list = [np.array(a) if not isinstance(a, np.ndarray) else a for a in a_list]

    # Validate that nominal_mean and nominal_cov are correct shapes
    if nominal_mean.shape != (L,):
        raise ValueError(f"nominal_mean must be a vector of length {L}.")
    if nominal_cov.shape != (L, L):
        raise ValueError(f"nominal_cov must be a {L}x{L} matrix.")

    # Ensure covariance matrix is positive semi-definite (for Gaussian distributions)
    if not np.all(np.linalg.eigvals(nominal_cov) >= 0):
        raise ValueError("nominal_cov must be positive semi-definite.")

    # Ensure epsilon is positive
    if epsilon <= 0:
        raise ValueError("epsilon must be a positive value.")

    # Ensure that the anomaly centers are sufficiently spaced to avoid ball overlap
    for i in range(len(a_list)):
        for j in range(i + 1, len(a_list)):
            distance = np.linalg.norm(a_list[i] - a_list[j])
            if distance < 2 * epsilon:
                raise ValueError(f"The balls around centers {a_list[i]} and {a_list[j]} overlap. "
                                 f"Ensure that the centers are at least {2 * epsilon} apart.")

    # Determine the number of anomalies
    n_anomalies = np.random.binomial(n_data, tau)
    if n_anomalies == 0:
        raise ValueError("There are no true anomalies. Try with a larger tau or n_data.")

    # Randomly permute indices and split into anomalies and nominals
    permute_indices = np.random.permutation(n_data)
    anomaly_indices = permute_indices[:n_anomalies]
    nominal_indices = permute_indices[n_anomalies:]

    # Initialize the data array (L dimensions)
    X = np.empty((n_data, L))

    # Determine how many anomalies each center should handle
    n_centers = len(a_list)
    n_per_center = n_anomalies // n_centers
    remainder = n_anomalies % n_centers

    anomalies_per_center = np.full(n_centers, n_per_center)
    for i in range(remainder):
        anomalies_per_center[i] += 1

    # Assign anomalies uniformly within the balls around each center
    start_idx = 0
    for i, center in enumerate(a_list):
        anomalies = []
        while len(anomalies) < anomalies_per_center[i]:
            # Generate candidate points uniformly within a cube around the center
            candidate_points = np.random.uniform(
                center - epsilon,
                center + epsilon,
                size=(anomalies_per_center[i] * 2, L)
            )
            # Retain only the points within the ball
            within_ball = candidate_points[
                np.linalg.norm(candidate_points - center, axis=1) <= epsilon
            ]
            anomalies.extend(within_ball[:anomalies_per_center[i] - len(anomalies)])

        # Add anomalies to the dataset
        X[anomaly_indices[start_idx:start_idx + anomalies_per_center[i]], :] = anomalies
        start_idx += anomalies_per_center[i]

    # Assign nominal points by sampling from the multivariate Gaussian distribution
    X[nominal_indices] = np.random.multivariate_normal(nominal_mean, nominal_cov, len(nominal_indices))

    # Create labels for anomalies and nominals
    Y = np.empty(n_data, dtype=int)
    Y[anomaly_indices] = 1
    Y[nominal_indices] = 0

    return X, Y

########################################################################################

def uniform_sampling_point_mass_with_epsilon_n_dim_rejection(n_data, tau, a_list, epsilon, L, lower=0, upper=1):
    """
    Generate nominal samples from a uniform distribution, with anomalies distributed
    around a set of centers defined by a_list. Anomalies are distributed in uniform
    volumes (L-dimensional cubes) around each center with the size of the volume determined by epsilon.
    
    Nominal points are uniformly distributed, but they are rejected if they fall inside
    any of the volumes around the anomaly centers.

    Parameters:
    - n_data: Number of samples.
    - tau: Fraction of the data which is an anomaly (0 <= tau <= 1).
    - a_list: List of L-dimensional centers for anomalies. Length of a_list defines the number of different anomaly volumes.
    - epsilon: The size of the volume around each center, determining the width of the anomaly distribution in each dimension.
    - lower: Lower bound of the range for nominal data (for validation).
    - upper: Upper bound of the range for nominal data (for validation).
    - L: Dimensionality of the space.

    Returns:
    - X: np.ndarray of sampled points (n_data x L).
    - Y: np.ndarray of corresponding labels (1 for anomalies, 0 for nominal points).
    """

    # Ensure a_list is not empty
    if len(a_list) == 0:
        raise ValueError("a_list must contain at least one value.")

    # Ensure that epsilon is a positive value
    if epsilon <= 0:
        raise ValueError("epsilon must be a positive value.")

    # Validate that the anomaly ranges do not exceed the bounds
    for center in a_list:
        if len(center) != L:
            raise ValueError(f"Each center in a_list must be a {L}-dimensional point.")

        # Ensure the box around the center does not exceed the bounds [lower, upper]
        if not all(lower <= center[i] - epsilon and center[i] + epsilon <= upper for i in range(L)):
            raise ValueError(f"Center {center} with epsilon {epsilon} creates a range outside of "
                             f"the bounds [{lower}, {upper}] for dimension {i}.")

    # Ensure that the volumes around the centers do not overlap
    for i in range(len(a_list)):
        for j in range(i + 1, len(a_list)):
            center_1 = np.array(a_list[i])
            center_2 = np.array(a_list[j])

            # Check overlap in each dimension
            overlap = all(
                abs(center_1[dim] - center_2[dim]) < 2 * epsilon
                for dim in range(L)
            )
            
            if overlap:
                raise ValueError(f"The volumes around centers {a_list[i]} and {a_list[j]} overlap. "
                                 f"Ensure that the centers are at least {2 * epsilon} apart in every dimension.")

    # Calculate the number of anomalies
    n_anomalies = np.random.binomial(n_data, tau)
    
    if n_anomalies == 0:
        raise ValueError("There are no true anomalies. Try with a larger tau or n_data.")

    # Randomly permute indices and split into anomalies and nominals
    permute_indices = np.random.permutation(n_data)
    anomaly_indices = permute_indices[:n_anomalies]
    nominal_indices = permute_indices[n_anomalies:]

    # Initialize the data array for X (with L-dimensional samples)
    X = np.empty((n_data, L))

    # Assign anomalies to corresponding centers in a_list
    start_idx = 0
    n_centers = len(a_list)
    n_per_center = n_anomalies // n_centers
    remainder = n_anomalies % n_centers

    # Handle any remainders in anomaly assignment
    anomalies_per_center = np.full(n_centers, n_per_center)
    for i in range(remainder):
        anomalies_per_center[i] += 1

    # Assign anomalies to corresponding centers in a_list
    for i, center in enumerate(a_list):
        lower_range = np.array(center) - epsilon
        upper_range = np.array(center) + epsilon

        # Sample the anomalies uniformly within the volume around each center
        X[anomaly_indices[start_idx:start_idx + anomalies_per_center[i]], :] = \
            np.random.uniform(lower_range, upper_range, size=(anomalies_per_center[i], L))

        start_idx += anomalies_per_center[i]

    # Now, let's generate nominal points with rejection sampling
    def is_inside_any_anomaly_box(point):
        """Check if the point falls inside any of the anomaly volumes."""
        for center in a_list:
            lower_range = np.array(center) - epsilon
            upper_range = np.array(center) + epsilon
            # Check if the point is inside the box for this anomaly center
            if np.all(lower_range <= point) and np.all(point <= upper_range):
                return True
        return False

    # Generate the nominal points by rejection sampling
    nominal_points = []
    while len(nominal_points) < len(nominal_indices):
        # Generate a random point uniformly
        point = np.random.uniform(lower, upper, L)

        # If the point is not inside any anomaly box, add it to the nominal points list
        if not is_inside_any_anomaly_box(point):
            nominal_points.append(point)

    # Convert the list of nominal points to a numpy array
    nominal_points = np.array(nominal_points)

    # Assign the nominal points to the appropriate indices
    X[nominal_indices] = nominal_points

    # Create labels for anomalies (1) and nominals (0)
    Y = np.empty(n_data, dtype=int)
    Y[anomaly_indices] = 1
    Y[nominal_indices] = 0

    return X, Y

########################################################################################

def multivariate_gaussian_sampling_with_epsilon_balls_rejection(n_data, tau, a_list, epsilon, nominal_mean, nominal_cov):
    """
    Generate nominal samples using a multivariate Gaussian distribution for nominals,
    with anomalies sampled uniformly within L-dimensional balls of radius epsilon
    around given centers. Nominals are generated using a rejection scheme so that they
    never fall within the balls around the anomaly centers.

    Parameters:
    - n_data: Number of samples.
    - tau: Fraction of the data which is an anomaly (0 <= tau <= 1).
    - a_list: List of L-dimensional centers for anomalies.
    - epsilon: Radius of the L-dimensional balls around each center for anomalies.
    - nominal_mean: L-dimensional mean vector for the Gaussian distribution of nominal data.
    - nominal_cov: LxL covariance matrix for the Gaussian distribution of nominal data.

    Returns:
    - X: np.ndarray of sampled points (size: n_data x L).
    - Y: np.ndarray of corresponding labels (1 for anomalies, 0 for nominal points).
    """

    # Ensure a_list is not empty
    if len(a_list) == 0:
        raise ValueError("a_list must contain at least one value.")

    # Set L from a_list[0]
    L = len(a_list[0])

    # Validate that a_list contains the correct number of dimensions
    for center in a_list:
        if len(center) != L:
            raise ValueError(f"Each element in a_list must be {L}-dimensional.")

    # Convert a_list (list of lists) into a list of numpy arrays inside the function
    a_list = [np.array(a) if not isinstance(a, np.ndarray) else a for a in a_list]

    # Validate that nominal_mean and nominal_cov are correct shapes
    if nominal_mean.shape != (L,):
        raise ValueError(f"nominal_mean must be a vector of length {L}.")
    if nominal_cov.shape != (L, L):
        raise ValueError(f"nominal_cov must be a {L}x{L} matrix.")

    # Ensure covariance matrix is positive semi-definite (for Gaussian distributions)
    if not np.all(np.linalg.eigvals(nominal_cov) >= 0):
        raise ValueError("nominal_cov must be positive semi-definite.")

    # Ensure epsilon is positive
    if epsilon <= 0:
        raise ValueError("epsilon must be a positive value.")

    # Ensure that the anomaly centers are sufficiently spaced to avoid ball overlap
    for i in range(len(a_list)):
        for j in range(i + 1, len(a_list)):
            distance = np.linalg.norm(a_list[i] - a_list[j])
            if distance < 2 * epsilon:
                raise ValueError(f"The balls around centers {a_list[i]} and {a_list[j]} overlap. "
                                 f"Ensure that the centers are at least {2 * epsilon} apart.")

    # Determine the number of anomalies
    n_anomalies = np.random.binomial(n_data, tau)
    if n_anomalies == 0:
        raise ValueError("There are no true anomalies. Try with a larger tau or n_data.")

    # Randomly permute indices and split into anomalies and nominals
    permute_indices = np.random.permutation(n_data)
    anomaly_indices = permute_indices[:n_anomalies]
    nominal_indices = permute_indices[n_anomalies:]

    # Initialize the data array (L dimensions)
    X = np.empty((n_data, L))

    # Determine how many anomalies each center should handle
    n_centers = len(a_list)
    n_per_center = n_anomalies // n_centers
    remainder = n_anomalies % n_centers

    anomalies_per_center = np.full(n_centers, n_per_center)
    for i in range(remainder):
        anomalies_per_center[i] += 1

    # Assign anomalies uniformly within the balls around each center
    start_idx = 0
    for i, center in enumerate(a_list):
        anomalies = []
        while len(anomalies) < anomalies_per_center[i]:
            # Generate candidate points uniformly within a cube around the center
            candidate_points = np.random.uniform(
                center - epsilon,
                center + epsilon,
                size=(anomalies_per_center[i] * 2, L)
            )
            # Retain only the points within the ball
            within_ball = candidate_points[
                np.linalg.norm(candidate_points - center, axis=1) <= epsilon
            ]
            anomalies.extend(within_ball[:anomalies_per_center[i] - len(anomalies)])

        # Add anomalies to the dataset
        X[anomaly_indices[start_idx:start_idx + anomalies_per_center[i]], :] = anomalies
        start_idx += anomalies_per_center[i]

    # Rejection scheme for nominal points: reject if inside any anomaly ball
    def is_inside_any_ball(point):
        for center in a_list:
            if np.linalg.norm(point - center) <= epsilon:
                return True
        return False

    # Assign nominal points by sampling from the multivariate Gaussian distribution with rejection
    nominal_points = []
    while len(nominal_points) < len(nominal_indices):
        candidate_point = np.random.multivariate_normal(nominal_mean, nominal_cov)
        if not is_inside_any_ball(candidate_point):
            nominal_points.append(candidate_point)

    X[nominal_indices] = np.array(nominal_points)

    # Create labels for anomalies and nominals
    Y = np.empty(n_data, dtype=int)
    Y[anomaly_indices] = 1
    Y[nominal_indices] = 0

    return X, Y

########################################################################################

def multivariate_gaussian_sampling_with_anomaly_gaussians(n_data, tau, a_list, anomaly_cov_list, nominal_mean, nominal_cov, L):
    """
    Generate nominal samples using a multivariate Gaussian distribution for nominals,
    with anomalies sampled from Gaussian distributions centered at given means and covariances.

    Parameters:
    - n_data: Number of samples.
    - tau: Fraction of the data which is an anomaly (0 <= tau <= 1).
    - a_list: List of L-dimensional means for anomalies.
    - anomaly_cov_list: List of LxL covariance matrices for anomalies.
    - nominal_mean: L-dimensional mean vector for the Gaussian distribution of nominal data.
    - nominal_cov: LxL covariance matrix for the Gaussian distribution of nominal data.
    - L: Dimensionality of the data.

    Returns:
    - X: np.ndarray of sampled points (size: n_data x L).
    - Y: np.ndarray of corresponding labels (1 for anomalies, 0 for nominal points).
    """
    
    # Validate that a_list contains the correct number of dimensions
    for center in a_list:
        if len(center) != L:
            raise ValueError(f"Each element in a_list must be {L}-dimensional.")
        
    # Ensure anomaly_cov_list contains numpy arrays and validate their shapes
    anomaly_cov_list = [
        np.array(cov) if not isinstance(cov, np.ndarray) else cov for cov in anomaly_cov_list
    ]
    for cov in anomaly_cov_list:
        if cov.shape != (L, L):
            raise ValueError(f"Each covariance matrix must be {L}x{L}.")
    
    # Ensure nominal covariance matrix is positive semi-definite (for Gaussian distributions)
    if not np.all(np.linalg.eigvals(nominal_cov) >= 0):
        raise ValueError("nominal_cov must be positive semi-definite.")
    
    # Ensure anomaly covariance matrices are positive semi-definite
    for cov in anomaly_cov_list:
        if not np.all(np.linalg.eigvals(cov) >= 0):
            raise ValueError("An anomaly covariance matrix is not positive semi-definite.")
    
    # Determine the number of anomalies
    n_anomalies = np.random.binomial(n_data, tau)
    if n_anomalies == 0 and tau != 0:
        raise ValueError("There are no true anomalies. Try with a larger tau or n_data.")
    
    # Randomly permute indices and split into anomalies and nominals
    permute_indices = np.random.permutation(n_data)
    anomaly_indices = permute_indices[:n_anomalies]
    nominal_indices = permute_indices[n_anomalies:]
    
    # Initialize the data array (L dimensions)
    X = np.empty((n_data, L))
    
    # Determine how many anomalies each center should handle
    n_centers = len(a_list)
    n_per_center = n_anomalies // n_centers
    remainder = n_anomalies % n_centers

    anomalies_per_center = np.full(n_centers, n_per_center)
    for i in range(remainder):
        anomalies_per_center[i] += 1

    # Assign anomalies by sampling from the Gaussian distributions
    start_idx = 0
    for i, (mean, cov) in enumerate(zip(a_list, anomaly_cov_list)):
        anomalies = np.random.multivariate_normal(mean, cov, anomalies_per_center[i])
        X[anomaly_indices[start_idx:start_idx + anomalies_per_center[i]], :] = anomalies
        start_idx += anomalies_per_center[i]

    # Assign nominal points by sampling from the multivariate Gaussian distribution
    X[nominal_indices] = np.random.multivariate_normal(nominal_mean, nominal_cov, len(nominal_indices))
    
    # Create labels for anomalies and nominals
    Y = np.empty(n_data, dtype=int)
    Y[anomaly_indices] = 1
    Y[nominal_indices] = 0
    
    return X, Y

########################################################################################

def sample_uniform_with_anomalies_in_ball(n_data, tau, a_list, epsilon, L, radius=1.0):
    """
    Generate samples uniformly within an L-dimensional ball (nominal data) 
    and smaller balls (anomalies) with rejection sampling for nominal points.
    
    Parameters:
    - n_data: Total number of samples.
    - tau: Fraction of anomalies (0 <= tau <= 1).
    - a_list: List of L-dimensional centers for anomaly balls. Length of a_list is the number of anomaly balls.
    - epsilon: Radius of each anomaly ball.
    - L: Dimensionality of the space.
    - radius: Radius of the large nominal ball.
    
    Returns:
    - X: np.ndarray of sampled points (n_data x L).
    - Y: np.ndarray of corresponding labels (1 for anomalies, 0 for nominal points).
    """
    def sample_point_in_ball(center, r, dim):
        """Samples a single point uniformly within an L-dimensional ball of radius r."""
        direction = np.random.normal(0, 1, dim)
        direction /= np.linalg.norm(direction)
        distance = r * (np.random.rand() ** (1 / dim))
        return center + direction * distance

    # Ensure the anomaly centers are valid
    for center in a_list:
        if len(center) != L:
            raise ValueError(f"Each center in a_list must be an {L}-dimensional point.")
        if np.linalg.norm(center) + epsilon > radius:
            raise ValueError(f"Anomaly center {center} with epsilon {epsilon} extends beyond the nominal ball radius.")

    # Ensure anomaly balls do not overlap
    for i in range(len(a_list)):
        for j in range(i + 1, len(a_list)):
            if np.linalg.norm(np.array(a_list[i]) - np.array(a_list[j])) < 2 * epsilon:
                raise ValueError(f"Anomaly balls around centers {a_list[i]} and {a_list[j]} overlap.")

    # Calculate the number of anomalies
    n_anomalies = np.random.binomial(n_data, tau)
    if n_anomalies == 0:
        raise ValueError("No anomalies generated. Increase tau or n_data.")

    # Initialize arrays
    X = np.empty((n_data, L))
    Y = np.empty(n_data, dtype=int)

    # Sample anomalies
    anomaly_indices = np.random.choice(n_data, n_anomalies, replace=False)
    nominal_indices = np.setdiff1d(np.arange(n_data), anomaly_indices)

    start_idx = 0
    for i, center in enumerate(a_list):
        n_per_center = n_anomalies // len(a_list) + (1 if i < n_anomalies % len(a_list) else 0)
        for j in range(n_per_center):
            X[anomaly_indices[start_idx]] = sample_point_in_ball(center, epsilon, L)
            Y[anomaly_indices[start_idx]] = 1
            start_idx += 1

    # Generate nominal points with rejection sampling
    def is_inside_any_anomaly(point):
        """Check if a point falls inside any anomaly ball."""
        for center in a_list:
            if np.linalg.norm(point - np.array(center)) < epsilon:
                return True
        return False

    nominal_points = []
    while len(nominal_points) < len(nominal_indices):
        point = sample_point_in_ball(np.zeros(L), radius, L)
        if not is_inside_any_anomaly(point):
            nominal_points.append(point)

    # Assign nominal points
    nominal_points = np.array(nominal_points)
    X[nominal_indices] = nominal_points
    Y[nominal_indices] = 0

    return X, Y

###################################################################################

def multivariate_gaussian_sampling_with_uniform_surface_anomalies(n_data, tau, radius, nominal_mean, nominal_cov, L=3):
    """
    Generate nominal samples using a multivariate Gaussian distribution for nominals,
    and anomalies uniformly on the surface of an L-dimensional sphere (or hypersphere) with a given radius.

    Parameters:
    - n_data: Number of samples.
    - tau: Fraction of the data which is an anomaly (0 <= tau <= 1).
    - radius: Radius of the hypersphere where anomalies will be located.
    - nominal_mean: L-dimensional mean vector for the Gaussian distribution of nominal data.
    - nominal_cov: LxL covariance matrix for the Gaussian distribution of nominal data.
    - L: Dimensionality of the data.

    Returns:
    - X: np.ndarray of sampled points (size: n_data x L).
    - Y: np.ndarray of corresponding labels (1 for anomalies, 0 for nominal points).
    """
    # Validate nominal_mean and nominal_cov shapes
    if nominal_mean.shape != (L,):
        raise ValueError(f"nominal_mean must be a vector of length {L}.")
    if nominal_cov.shape != (L, L):
        raise ValueError(f"nominal_cov must be a {L}x{L} matrix.")
    
    # Ensure covariance matrix is positive semi-definite
    if not np.all(np.linalg.eigvals(nominal_cov) >= 0):
        raise ValueError("nominal_cov must be positive semi-definite.")

    # Determine the number of anomalies
    n_anomalies = int(np.round(n_data * tau))
    if n_anomalies == 0:
        raise ValueError("There are no true anomalies. Try with a larger tau or n_data.")
    
    # Determine the number of nominal samples
    n_nominals = n_data - n_anomalies
    
    # Sample nominal points from the multivariate Gaussian distribution
    X_nominals = np.random.multivariate_normal(nominal_mean, nominal_cov, n_nominals)
    
    # Generate anomalies uniformly on the surface of a hypersphere
    # Step 1: Sample from a standard normal distribution
    anomalies = np.random.normal(size=(n_anomalies, L))
    
    # Step 2: Normalize to unit length (ensures uniform distribution)
    norms = np.linalg.norm(anomalies, axis=1, keepdims=True)
    anomalies = anomalies / norms  # Points are now on the surface of a unit sphere
    
    # Step 3: Scale anomalies to the desired radius
    anomalies *= radius

    # Combine nominal and anomaly points
    X = np.vstack((X_nominals, anomalies))
    
    # Create labels: 0 for nominals, 1 for anomalies
    Y = np.zeros(n_data, dtype=int)
    Y[n_nominals:] = 1  # Last n_anomalies points are anomalies

    # Shuffle data to mix anomalies and nominals
    indices = np.random.permutation(n_data)
    X = X[indices]
    Y = Y[indices]
    
    return X, Y
