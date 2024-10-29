# Linear Regression

## Principle

Using a linear function to fit the data, calculate the loss using **Mean Squared Error (MSE)**, and then use **Gradient Descent (GD)** to find a set of weights that minimize the MSE.

## Derivation of Linear Regression

### 1. Hypothesis

The hypothesis function is given as:

$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2
$$

or more generally,

$$
h_\theta(x) = \theta^T x = \sum_{i=0}^n \theta_i x_i
$$

### 2. Error Term

The difference between the actual value and the predicted value introduces an error term \( $\epsilon$ \):

$$
y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}
$$

The error term is assumed to be independently and follow a normal distribution with mean 0 and variance \$\sigma^2$ \:

$$
p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma} \exp \left( - \frac{({\epsilon^{(i)})}^2}{2\sigma^2} \right)
$$

### 3. Likelihood Function

The probability of observing the data \($y^{(i)}$\) given the input \($x^{(i)}$\) and parameters \($\theta$\), based on the assumption of normal errors, is given by:

$$
p(y^{(i)} | x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp \left( - \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)
$$

For all data points, assuming independence of errors, the joint probability (likelihood) is the product of these individual probabilities:


$$
L(\theta) = \prod_{i=1}^m  p(y^{(i)} | x^{(i)}; \theta) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma} \exp \left( - \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)
$$

### 4. Log-Likelihood

To simplify the maximization process, we take the logarithm of the likelihood function (log-likelihood):

$$
\log L(\theta) = \sum_{i=1}^m \log \left( \frac{1}{\sqrt{2\pi}\sigma} \exp \left( - \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right) \right)
$$

This simplifies to:

$$
\log L(\theta) = m \log(\frac{1}{\sqrt{2\pi}\sigma}) - \frac{1}{2\sigma^2} \sum_{i=1}^m (y^{(i)} - \theta^T x^{(i)})^2
$$

Since the first term is constant, maximizing the log-likelihood is equivalent to minimizing the following term:

$$
J(\theta) = \frac{1}{2} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2
$$

### 5. Gradient Descent

To minimize the cost function \$J(\theta)$ \, we take its gradient with respect to \$\theta$ \:

$$
\nabla J(\theta) = X^T X \theta - X^T y
$$

Setting the gradient to zero gives the normal equation:

$$
\theta = (X^T X)^{-1} X^T y
$$

This is the solution for the parameters \( $\theta$ \) that minimize the MSE.
