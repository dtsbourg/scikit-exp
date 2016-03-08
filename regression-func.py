# First usage of scikit-learn
# Based on http://nbviewer.jupyter.org/github/ipython-books/cookbook-code/tree/master/notebooks/
# Goal : Use regression to estimate a known function

import numpy as np
import scipy.stats as st
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

f = lambda x: np.exp(3 * x)

x_t = np.linspace(0., 2, 200)
y_t = f(x_t)

# Generate training data (with gaussian noise)
x = np.linspace(0., 2, 200)
y = f(x) + 5 * np.random.randn(len(x))


def base_model():
    """ Showing the function to estimate and sampled data points """
    # Plot
    plt.figure(figsize=(6,3))
    plt.plot(x_t[:200], y_t[:200], '--k')
    plt.plot(x, y, 'ok', ms=10)
    plt.show()

def lr():
    """ Linear Regression """
    # Create model
    lr = lm.LinearRegression()
    # Train model on our training dataset
    lr.fit(x[:, np.newaxis], y) # newaxis adds a dimension to x (now 200x1) to make it a col vector
    # Predict points with our trained model
    y_lr = lr.predict(x_t[:, np.newaxis])

    plt.figure(figsize=(6,3))
    plt.plot(x_t, y_t, '--k')
    plt.plot(x_t, y_lr, 'g')
    plt.plot(x, y, 'ok', ms=10)
    plt.xlim(0, 2)
    plt.ylim(y.min()-1, y.max()+1)
    plt.title("Linear regression")
    plt.show()

def lr_plot(lrp):
    plt.figure(figsize=(6,3))
    plt.plot(x_t, y_t, '--k')

    for deg, s in zip([2, 5], ['-', '.']): # Just a classy way to plot degree 2 with -- and degree 5 with .
        lrp.fit(np.vander(x, deg + 1), y)
        y_lrp = lrp.predict(np.vander(x_t, deg + 1))
        plt.plot(x_t, y_lrp, s, label='degree ' + str(deg))
        plt.legend(loc=2)
        plt.xlim(x.min()-0.1, x.max()+0.1)
        plt.ylim(y.min()-1, y.max()+1)
        plt.plot(x, y, 'ok', ms=10)
        plt.title("Linear regression")
    plt.show()

def lr_vander():
    lrp = lm.LinearRegression()
    lr_plot(lrp)

def lr_ridge():
    """
    Prevents fitting coefficients to grow too big (as is the case with
    regular polynomial linear regression) by adding a regularization term.
    Effectively it is a cost function that minimizes weights as well as error
    between model and data
    """
    lrp = lm.RidgeCV()
    lr_plot(lrp)

def main():
    lr_ridge()


if __name__ == "__main__":
    main()
