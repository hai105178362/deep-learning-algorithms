# 11785 Week2 Notes

## Difference Between Perceptron, Adaline and Neural Network Models

Both Adaline and the Perceptron are (single-layer) neural network models. The Perceptron is one of the oldest and simplest learning algorithms out there, and I would consider Adaline as an improvement over the Perceptron.

### What Adaline and the Perceptron have in common
* they are classifiers for binary classification
* both have a linear decision boundary
* both can learn iteratively, sample by sample (the Perceptron naturally, and Adaline via stochastic gradient descent)
* both use a threshold function