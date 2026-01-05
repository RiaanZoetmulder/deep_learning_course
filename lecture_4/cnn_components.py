import numpy as np
from mnist_datasets import MNISTLoader

#### Feedforward Layer
class FFNLayer():
  """
  A custom implementation of a Feedforward Neural Network (FFN) layer.

  This class implements a standard dense layer with learnable weights and
  provides methods for the forward and backward passes, as well as weight updates.
  It includes support for different weight initialization schemes and gradient clipping.
  """
  def __init__(self, in_dims:int, out_dims: int, initialization: str = 'normal'):
    """
    Initializes the FFNLayer.

    Parameters
    ----------
    in_dims : int
        The number of input dimensions (number of neurons in the previous layer).
    out_dims : int
        The number of output dimensions (number of neurons in this layer).
    initialization : str, optional
        The weight initialization method to use. Supported options are 'normal'
        (random normal distribution) and 'kaiming' (Kaiming He initialization).
        Defaults to 'normal'.
    """
    self.in_dims = in_dims
    self.out_dims = out_dims
    self.gradient_clip_threshold: float = 1.0 

    if initialization == 'normal':
      self.W = np.random.randn(in_dims, out_dims)

    elif initialization == 'kaiming':
      std_dev = np.sqrt(2.0 / in_dims) 
      self.W = np.random.randn(in_dims, out_dims) * std_dev

    self._learning_rate = 1.0

  @property
  def learning_rate(self) -> float:
    return self._learning_rate

  @learning_rate.setter
  def learning_rate(self, value: float) -> None:
    self._learning_rate = value

  def __call__(self, X: np.ndarray) -> np.ndarray:
    """
    Performs the forward pass of the FFN layer.

    Calculates the output of the layer by performing a matrix multiplication
    of the input tensor with the layer's weights.

    Parameters
    ----------
    X : np.ndarray
        Input tensor to the FFN layer. Expected shape (Batch, in_dims).

    Returns
    -------
    np.ndarray
        Output tensor of the FFN layer. Shape (Batch, out_dims).
    """
    self.X = X
    out = X @ self.W

    return out

  def backpropagate(self, delta: np.ndarray) -> np.ndarray:
    """
    Performs the backward pass of the FFN layer.

    Calculates the gradients for the weights, updates them, and computes
    the gradient with respect to the input to pass back to the previous layer.

    Parameters
    ----------
    delta : np.ndarray
        Gradient of the loss with respect to the output of this layer.
        Shape (Batch, out_dims).

    Returns
    -------
    np.ndarray
        Gradient of the loss with respect to the input of this layer.
        Shape (Batch, in_dims).
    """

    self.update_weights(delta)

    # Delta next is the derivative wrt the input of this network
    delta_next = delta @ self.W.T

    return delta_next

  def update_weights(self, delta: np.ndarray) -> None:
    """
    Updates the weights of the FFN layer using calculated gradients
    and the learning rate, with gradient clipping.
    """

    # Calculate the raw gradient for weights
    gradients_W = self.X.T @ delta

    # Apply gradient clipping to weights
    weight_grad_norm = np.linalg.norm(gradients_W)
    if weight_grad_norm > self.gradient_clip_threshold:
        gradients_W = gradients_W * (self.gradient_clip_threshold / weight_grad_norm)


    self.W = self.W - self._learning_rate * gradients_W


class LeakyReLU():
  """
  Implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function.

  Leaky ReLU is a variation of the ReLU activation function that allows a small
  gradient when the input is negative, addressing the "dying ReLU" problem.
  """
  def __init__(self, alpha: float = 0.01):
    """
    Initializes the LeakyReLU layer.

    Parameters
    ----------
    alpha : float, optional
        The slope of the activation function for negative inputs.
        Defaults to 0.01.
    """
    self.alpha = alpha
    self.X = None

  def __call__(self, X: np.ndarray) -> np.ndarray:
    """
    Performs the forward pass of the LeakyReLU layer.

    Applies the Leaky ReLU function to the input tensor X.

    Parameters
    ----------
    X : np.ndarray
        Input tensor to the LeakyReLU layer. Can have any shape.

    Returns
    -------
    np.ndarray
        Output tensor after applying Leaky ReLU. Has the same shape as the input.
    """
    self.X = X
    return np.where(self.X > 0, self.X, self.alpha * self.X)

  def backpropagate(self, delta: np.ndarray) -> np.ndarray:
    """
    Performs the backward pass of the LeakyReLU layer.

    Calculates the gradient with respect to the input X based on the
    Leaky ReLU derivative.

    Parameters
    ----------
    delta : np.ndarray
        Gradient of the loss with respect to the output of this layer.
        Has the same shape as the output.

    Returns
    -------
    np.ndarray
        Gradient of the loss with respect to the input of this layer.
        Has the same shape as the input.
    """
    binarized_matrix = np.where(self.X > 0, 1, self.alpha)
    return delta * binarized_matrix


class SoftMax():
  """
  Implements the SoftMax activation function.

  The SoftMax function converts a vector of numbers into a probability
  distribution, where the sum of the probabilities is 1. It is often used
  as the last layer in a neural network for multi-class classification tasks.
  """
  def __init__(self):
    """
    Initializes the SoftMax layer.

    This layer does not have any learnable parameters.
    """
    self.X = None
    pass

  def __call__(self, X: np.ndarray):
    """
    Performs the forward pass of the SoftMax layer.

    Applies the SoftMax function to the input tensor X. For numerical stability,
    the maximum value in each input row is subtracted before exponentiation.

    Parameters
    ----------
    X : np.ndarray
        Input tensor to the SoftMax layer. Expected shape (Batch, num_classes).

    Returns
    -------
    np.ndarray
        Output tensor with the SoftMax probabilities. Shape (Batch, num_classes).
    """
    # Numerically stable softmax: subtract the maximum value for stability
    exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

  def backpropagate(self, delta: np.ndarray) -> np.ndarray:
    """
    Performs the backward pass of the SoftMax layer.

    Assuming the loss function is Cross-Entropy, the gradient with respect
    to the input X is simply the gradient from the next layer (delta).

    Parameters
    ----------
    delta : np.ndarray
        Gradient of the loss with respect to the output of this layer.
        Shape (Batch, num_classes).

    Returns
    -------
    np.ndarray
        Gradient of the loss with respect to the input of this layer.
        Shape (Batch, num_classes).
    """
    return delta

class CrossEntropy():
  def __init__(self, epsilon: float = 1e-8):
    self.epsilon = epsilon
    self.Y_true = None

  def __call__(self, Y_True: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    self.Y_true = Y_True
    return -np.sum(Y_True * np.log(Y_pred + self.epsilon))

  def backpropagate(self, Y_pred: np.ndarray) -> np.ndarray:
    return Y_pred - self.Y_true

class Flatten():
    """
    A simple Flatten layer for reshaping the input tensor.

    This layer is typically used between convolutional/pooling layers
    and fully connected (feedforward) layers to convert the multi-dimensional
    output of the convolutional part into a 2D tensor that can be processed
    by the feedforward layers.
    """
    def __init__(self):
        """
        Initializes the Flatten layer.

        This layer does not have any learnable parameters.
        """
        self.original_shape = None # To store the shape of the input for the backward pass

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the Flatten layer.

        Reshapes the input tensor from (Batch, ...) to (Batch, flattened_dimensions).

        Parameters
        ----------
        X : np.ndarray
            Input tensor to the Flatten layer. Can have any shape.

        Returns
        -------
        np.ndarray
            Flattened output tensor with shape (Batch, flattened_dimensions).
        """
        # Store the original shape of the input tensor for the backward pass
        self.original_shape = X.shape
        B = X.shape[0]

        # Calculate the total number of elements after the batch dimension
        flattened_size = np.prod(X.shape[1:])
        output_tensor = X.reshape(B, flattened_size)

        return output_tensor

    def backpropagate(self, delta: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the Flatten layer.

        Reshapes the gradient tensor from (Batch, flattened_dimensions)
        back to the original input shape.

        Parameters
        ----------
        delta : np.ndarray
            Gradient of the loss with respect to the output of this layer.
            Shape (Batch, flattened_dimensions).

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the input of this layer,
            reshaped back to the original input shape.
            Shape self.original_shape.
        """

        output_delta = delta.reshape(self.original_shape)
        return output_delta


class DataLoader():
  def __init__(self, batch_size = 8, X: np.ndarray = None, y: np.ndarray = None, flatten: bool = False):
    '''
    Simple DataLoader, PyTorch style!

    Don't modify anything here!
    '''
    self.X = X
    if not flatten:
      self.X = X.reshape(X.shape[0], 1, 28, 28)

    self.y = y
    self.shuffle_data()
    self.batch_size = batch_size

    self.start_index = 0
    self.end_index = self.batch_size

  def shuffle_data(self):
    indices = np.arange(self.X.shape[0])
    np.random.shuffle(indices)
    self.X = self.X[indices]
    self.y = self.y[indices]

  def __len__(self):
    return len(self.X)

  def __iter__(self):
    return self

  def __next__(self):

    # reset indices and shuffle data
    if self.end_index > self.X.shape[0]:
      self.start_index = 0
      self.end_index = self.batch_size
      self.shuffle_data()
      raise StopIteration

    next_X = self.X[self.start_index:self.end_index]
    next_y = self.y[self.start_index:self.end_index]

    # Ensure indices are updated
    self.start_index = self.end_index
    self.end_index += self.batch_size

    return next_X, next_y

# class for the MNIST dataloader. DO NOT CHANGE THIS.
class MNISTDataLoaderFactory():
  def __init__(self, batch_size: int  = 8, flatten: bool = True) -> None:
    '''
    Simple Dataloader, PyTorch style!

    Don't modify anything here!
    '''
    train_X, train_y = MNISTLoader().load()
    validate_X, validate_y = MNISTLoader().load(train = False )

    # Calculate mean and standard deviation, add epsilon for stability
    mean_train = np.mean(train_X, axis = 0)
    std_train = np.std(train_X)  # Add epsilon
    mean_validate = np.mean(validate_X, axis=0)
    std_validate = np.std(validate_X) # Corrected typo

    # normalize the features
    train_X = (train_X - mean_train) / std_train
    validate_X = (validate_X - mean_validate) / std_validate

    # convert y_train and y_validate into one_hot matrix
    train_y  = np.array(train_y).astype(int)
    validate_y  = np.array(validate_y).astype(int)

    train_y_one_hot  = np.zeros((train_y.shape[0], 10))
    train_y_one_hot[np.arange(train_y.shape[0]), train_y] = 1.

    validate_y_one_hot  = np.zeros((validate_y.shape[0], 10))
    validate_y_one_hot[np.arange(validate_y.shape[0]), validate_y] = 1.

    self.batch_size = batch_size

    self.train_dataset = DataLoader(self.batch_size, train_X, train_y_one_hot, flatten = flatten)
    self.validation_dataset = DataLoader(self.batch_size, validate_X, validate_y_one_hot, flatten = flatten)

    self.len_train = len(self.train_dataset)
    self.len_validate = len(self.validation_dataset)

  def get_validation_dataset(self):
    return self.validation_dataset

  def get_train_dataset(self):
    return self.train_dataset

class ModuleList():
  """
  A container for neural network layers, similar to PyTorch's nn.ModuleList.

  This class allows for sequential composition of layers and facilitates
  forward and backward passes through the entire network.
  """
  def __init__(self):
    self.layers = []

  def set_learning_rate(self, lr: float = 0.01):
    """
    Sets the learning rate for all layers that have a 'learning_rate' attribute.

    Parameters
    ----------
    lr : float, optional
        The learning rate to set. Defaults to 0.01.
    """
    for layer in self.layers:
      if hasattr(layer, 'learning_rate'):
        layer.learning_rate = lr

  def set_training(self):
    """
    Sets all layers with a 'training' attribute to training mode.
    """
    for layer in self.layers:
      if hasattr(layer, 'training'):
        layer.training = True

  def set_evaluation(self):
    """
    Sets all layers with a 'training' attribute to evaluation mode.
    """
    for layer in self.layers:
      if hasattr(layer, 'training'):
        layer.training = False

  def add(self, layer):
    """
    Adds a layer to the ModuleList.

    Parameters
    ----------
    layer : object
        The layer object to add. It should have a __call__ method for
        the forward pass and a backpropagate method for the backward pass.
    """
    self.layers.append(layer)

  def __call__(self, X: np.ndarray) -> np.ndarray:
    """
    Performs the forward pass through all layers in the ModuleList.

    Parameters
    ----------
    X : np.ndarray
        The input tensor to the first layer.

    Returns
    -------
    np.ndarray
        The output tensor after passing through all layers.
    """
    # NOTE: This is how they forward pass will be called
    # for all of modules.
    for layer in self.layers:
      X = layer(X)
    return X

  def backpropagate(self, delta: np.ndarray) -> np.ndarray:
    """
    Performs the backward pass through all layers in the ModuleList in reverse order.

    Parameters
    ----------
    delta : np.ndarray
        The gradient of the loss with respect to the output of the last layer.

    Returns
    -------
    np.ndarray
        The gradient of the loss with respect to the input of the first layer.
    """
    for layer in reversed(self.layers):
      delta = layer.backpropagate(delta)
    return delta
