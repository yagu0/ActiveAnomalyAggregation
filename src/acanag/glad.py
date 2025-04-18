import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

def build_neural_network(data_dimension, M, l2_lambda=0.01):

    """
    Builds a neural network with the required architecture for the GLAD algorithm.
    
    Md Rakibul Islam, Shubhomoy Das, Janardhan Rao Doppa, and Sriraam Natarajan.
    Glad: Glocalized anomaly detection via human-in-the-loop learning. arXiv preprint
    arXiv:1810.01403, 2018.

    Parameters:
    - data_dimension (int): Number of features (columns) in the input data.
    - M (int): Number of neurons in the output layer.
    - l2_lambda: neural network parameter

    Returns:
    - model (tf.keras.Model): The constructed neural network.
    """

    def custom_sigmoid(x):
        return tf.keras.activations.sigmoid(x)
    
    # Define the number of neurons in the hidden layer (specified in the GLAD paper)
    hidden_neurons = max(50, M * 3)

    # Create the model
    model = Sequential([
        # Input layer (using Input layer)
        Input(shape=(data_dimension,)),  # Define the input shape explicitly
        
        # Hidden layer with LeakyReLU activation
        Dense(hidden_neurons, activation=None,kernel_regularizer=regularizers.l2(l2_lambda)),
        
        # Output layer with sigmoid activation
        Dense(M, activation=custom_sigmoid,kernel_regularizer=regularizers.l2(l2_lambda))
    ])

    return model

def custom_binary_crossentropy_loss(b=0.5,mylambda = 1):
    
    """
    Custom binary cross-entropy loss function where the target for each output
    node is a constant value b. This is the initialization loss described in
    Islam et al.

    Md Rakibul Islam, Shubhomoy Das, Janardhan Rao Doppa, and Sriraam Natarajan.
    Glad: Glocalized anomaly detection via human-in-the-loop learning. arXiv preprint
    arXiv:1810.01403, 2018.

    Remarks: (1) y_true is just a place-holder here since we actually just use a 
            vector of b's in its place for this specific, initialization, loss.
             (2) y_pred is the outputs of the neural network (== p_m(x))

    Parameters:
    - b: The target value for each output (between 0 and 1). Default is 0.5.
    - mylambda: see Islam et al. for details. Default is 1.

    Returns:
    - loss: A loss function that can be used for model compilation.
    """
    
    def loss(y_true, y_pred):
        # Ensure that y_pred is in the range (0, 1)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Binary cross entropy term for each output
        term1 = b * tf.math.log(y_pred)
        term2 = (1 - b) * tf.math.log(1 - y_pred)
        
        # Calculate the total loss over all output neurons
        loss = -tf.reduce_sum(term1 + term2, axis=-1)

        # Normalize by the batch size (number of data points in the current batch)
        batch_size = tf.shape(y_pred)[0]  # Number of data points in the batch
        loss = mylambda * loss / tf.cast(batch_size, tf.float32)  # Divide by batch size

        return loss

    return loss

def new_custom_loss(X_lab, Y_lab, q_tau_tm1, all_labeled_scores, model, X_so_far, mylambda=1, b=0.5):
    
    """
    Unified loss function that combines the two terms described in Islam et al. 

    Md Rakibul Islam, Shubhomoy Das, Janardhan Rao Doppa, and Sriraam Natarajan.
    Glad: Glocalized anomaly detection via human-in-the-loop learning. arXiv preprint
    arXiv:1810.01403, 2018.

    Parameters:
    - X_lab: 2D NumPy array of shape (n_labeled, d).
    - Y_lab: 1D NumPy array of length n_labeled.
    - q_tau_tm1: Scalar value for the threshold parameter.
    - all_labeled_scores: 2D NumPy array of shape (n_labeled, M), corresponding 
        to raw scores of labeled data so far.
    - model: the neural network model from build_neural_network()
    - X_so_far: 2D NumPy array of shape (n_so_far, d).
    - mylambda: Scalar weight for the second term.
    - b: Target value used in the custom_binary_crossentropy_loss() function.

    Returns:
    - loss: The computed loss as a TensorFlow scalar.
    """
    
    # Calculate the first term
    n_labeled = X_lab.shape[0]
    term1_sum = 0.0

    Y_lab_internal = [2*Y_lab[m] - 1 for m in range(n_labeled)]
    
    # Iterate over all rows of X_labeled
    for r in range(n_labeled):
        w = model(X_lab[r:r+1])  # Neural network outputs for row r
        scalar_product = tf.reduce_sum(w * all_labeled_scores[r])
        term1_sum += tf.maximum(0.0, Y_lab_internal[r] * (q_tau_tm1 - scalar_product))
    
    first_term = term1_sum / n_labeled

    # Calculate the second term using the earlier custom_binary_crossentropy_loss
    n_so_far = X_so_far.shape[0]
    outputs = model(X_so_far)  # Outputs for X_so_far from the model
    binary_loss_fn = custom_binary_crossentropy_loss(b, mylambda)
    second_term = tf.reduce_sum(binary_loss_fn(None, outputs)) / n_so_far
    second_term *= mylambda  # Scale by lambda

    # Combine the terms to calculate the final loss
    total_loss = first_term + second_term

    return total_loss
