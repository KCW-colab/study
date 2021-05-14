import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
    # Compute f = Wx
    f = np.dot(X[i], W)
    # shift value of f for numeric stabliity 
    f -= np.max(f)

    scores = np.exp(f) / np.sum(np.exp(f))
    loss += -np.log(scores[y[i]] + 1e-7)
    # Weight Gradient
    for j in range(num_class):
      dW[:, j] += X[i] * scores[j]
    dW[:, y[i]] -= X[i]

  # Average
  loss /= num_train
  dW /= num_train

  # L2 - Regularization
  loss += reg * np.sum(W*W)
  dW +=  2 * reg * W
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_class = W.shape[1]

  f = np.dot(X, W)
  f -= np.max(f, axis = 1, keepdims = True).reshape(-1,1)
  
  score = np.exp(f) / np.sum(np.exp(f), axis = 1, keepdims = True)
  loss += np.sum( -np.log(score[range(num_train), y] + 1e-7))

  # Average
  loss /= num_train
  
  # L-2 Regularization
  loss += reg * np.sum(W * W)

  ind = np.zeros_like(score)
  ind[np.arange(num_train), y] = 1
  dW = X.T.dot(score - ind)

  
  dW = dW / num_train + 2 * reg * W 
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

