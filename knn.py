# An example using k-NN for classification
# based on * http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html#k-nearest-neighbors-classifier
#          * http://www.scipy-lectures.org/packages/scikit-learn/index.html#k-nearest-neighbors-classifier

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

k = 15   # number of neighbors :  larger k suppresses the effects of noise, but makes the classification boundaries less distinct
h = .02  # step size in the mesh
n_samples = 50

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# load the dataset
iris = datasets.load_iris()
iris_x = iris.data[:, :2]
iris_y = iris.target

# Randomly seperate data into training and testing sets
np.random.seed(0)
indices = np.random.permutation(len(iris_x))
iris_X_train = iris_x[indices[:-n_samples]]
iris_y_train = iris_y[indices[:-n_samples]]
iris_X_test  = iris_x[indices[-n_samples:]]
iris_y_test  = iris_y[indices[-n_samples:]]

# Train and plot k-NN for :
#   * Uniform weights : uniform weights to each neighbor
#   * Distance weights : weights proportional to the inverse of the distance from the query point
for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(k, weights=weights)
    clf.fit(iris_X_train, iris_y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = iris_X_train[:, 0].min() - 1, iris_X_train[:, 0].max() + 1
    y_min, y_max = iris_X_train[:, 1].min() - 1, iris_X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    score = clf.score(iris_X_test, iris_y_test)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(iris_X_train[:, 0], iris_X_train[:, 1], c=iris_y_train, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s') ; Score = %s"
              % (k, weights, score))

plt.show()
