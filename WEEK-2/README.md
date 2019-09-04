# Lecture 3 (9/4)

## Recap

- Neural Network can model any:
  - Boolean function
  - Classification boundary
  - Continuous valued function
- Networks with fewer than required parameters can be very poor approximator.

## Perceptron

<a href="https://www.codecogs.com/eqnedit.php?latex=Z=\sum_iw_ix_i&plus;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z=\sum_iw_ix_i&plus;b" title="Z=\sum_iw_ix_i+b" /></a>

- Inputs are real value
- `b` is the bias term, representing a threshold trigger the perceptron
- Activation function are `not` necessarily threshold functions.

### MLP

- MLP **can** represent anything

### Algorithm

- Given N training instance, <a href="https://www.codecogs.com/eqnedit.php?latex=$(X_1,Y_1),(X_2,Y_2),...,(X_N,Y_N)$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$(X_1,Y_1),(X_2,Y_2),...,(X_N,Y_N)$" title="$(X_1,Y_1),(X_2,Y_2),...,(X_N,Y_N)$" /></a>, where `Y = +1` or `Y=-1`
- Initialize W
- Cycle through the training dataset:
- Do:

```
for i = 1...N_Train:
    O(x_i) = Sign(W^TX_i)
    if (O(x_i) != y_i):
        W = W+y_i*x_i
    until no more classification errors
```

### Convergence of Perceptron

- **Guaranteed** to converge if classes are linearly separable
- After no more than <a href="https://www.codecogs.com/eqnedit.php?latex=({R\over\gamma})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?({R\over\gamma})^2" title="({R\over\gamma})^2" /></a> misclassifications
  - Specifically when `W` is set to 0
- R: length of longest training position
- <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" /></a>: the best case closest distance of a training point from the classifier.
  - Same as margin in SVM

## More Complex Decision Boundaries

- Direct perceptron algorithm can not be used for MLP since the time complexity is **exponential**
- The simple MLP is a flat, non-differentiable function
- In real life, data are mostly not linearly separable, Rosenblatt’s perceptron wouldn’t work in the first place.

### Perceptrons with differentiable activation functions

<a href="https://www.codecogs.com/eqnedit.php?latex=$$y^k_j=\sigma(\sum{w_{i,j}^{k-1},y_i^{k-1}})$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$y^k_j=\sigma(\sum{w_{i,j}^{k-1},y_i^{k-1}})$$" title="$$y^k_j=\sigma(\sum{w_{i,j}^{k-1},y_i^{k-1}})$$" /></a>

#### Expected Error/Risk

The empirical estimate of the expected error is the average error over the samples

<a href="https://www.codecogs.com/eqnedit.php?latex=$$E[div(f(X;W),g(X))]\approx&space;{1\over&space;N}\sum{div(f(X_i,;W),d_i)}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$E[div(f(X;W),g(X))]\approx&space;{1\over&space;N}\sum{div(f(X_i,;W),d_i)}$$" title="$$E[div(f(X;W),g(X))]\approx {1\over N}\sum{div(f(X_i,;W),d_i)}$$" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=$$Loss(W)={1\over&space;N}\sum{div(f(X_i,;W),d_i)}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$Loss(W)={1\over&space;N}\sum{div(f(X_i,;W),d_i)}$$" title="$$Loss(W)={1\over N}\sum{div(f(X_i,;W),d_i)}$$" /></a>




