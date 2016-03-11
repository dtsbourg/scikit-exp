# An example using a one-class SVM for novelty detection.
# based on http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#example-svm-plot-oneclass-py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV


# Generate problem data
############### SVM ###############
# Create a mesh of size (len(y), len(x)) to evaluate the function on
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500)) # np.linspace(start, stop, num)

# Generate training data
X = 0.3 * np.random.randn(200, 2) # 100 (x,y) random coordinates in [0;1]
# Center the data around (2,2) and (-2,-2) which will be the "nucleus" for the learned frontiers
X_train = np.r_[X + 2, X, X - 2] # np.r_ concatenates the slices

# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X, X - 2]

# Generate some abnormal novel observations (away from (2,2) and (-2,-2))
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

############### SVC ###############
X_svc_outliers = np.random.uniform(low=-4, high=4, size=(50, 2))
X = np.vstack((X_svc_outliers, X_train))
Y = np.append(-1 * np.ones(len(X_svc_outliers)), np.ones(len(X_train)))

def svm_fit(kernel):
    """
    Fit the model with a Support Vector Machine
     OneClassSVM :
        - nu : An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors
        - kernel : similarity function - takes two inputs and spits out how similar they are
        - gamma : Kernel coefficient (default = 1/n_features) ~ how far the influence of a single training example reaches

     Kernels :
        - rbf : a function such that phi(x) = phi(||x||)
        - linear
        - poly
        - sigmoid
    """
    clf = svm.OneClassSVM(nu=0.1, kernel=kernel, gamma=0.4)
    clf.fit(X_train) # Fit the model
    # Predict : perform regression on samples in X and return expected value (+1, -1 for one class example)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    # Find the number of errors
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    return clf

def svc_fit(optimal):
    """
    Fit the model with a Support Vector Classifier
    """
    clf = svm.SVC(C=optimal['C'], kernel='rbf', gamma=optimal['gamma'])
    clf.fit(X, Y)
    return clf

def svc_optimal_params():
    """
    Find the optimal set of parameters for the SVC given data X and labels Y

    Available params :
        ['kernel', 'C', 'verbose', 'probability', 'degree', 'shrinking',
         'max_iter', 'decision_function_shape', 'random_state', 'tol',
         'cache_size', 'coef0', 'gamma', 'class_weight']
    """
    C_range = np.logspace(-2, 2, 5)
    gamma_range = np.logspace(-2, 2, 5)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(Y, n_iter=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, Y)

    print("The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))
    return grid.best_params_

def svm_plot(decision_function):
    """ Plot results """
    # plot the line, the points, and the nearest vectors to the plane
    Z = decision_function(np.c_[xx.ravel(), yy.ravel()]) # returns the distance to the decision function border
    # ravel() flattens the array (eg 5x5 -> 25x1 )
    # c_ concatenates along the second axis : (25x1) -> (25x2)
    Z = Z.reshape(xx.shape) # ?

    plt.title("Novelty Detection")
    # Filled 2D contour function
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
    # Learned frontier
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    # Fill the nominal area
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

    # Data points
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')    # Training observations
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')      # New nominal observations
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red') # New anomalies
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a.collections[0], b1, b2, c],
               ["learned frontier", "training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    # plt.xlabel(
    #    "error train: %d/200 ; errors novel regular: %d/40 ; "
    #    "errors novel abnormal: %d/40"
    #    % (n_error_train, n_error_test, n_error_outliers))
    plt.show()

def svm_run():
    clf = svm_fit('rbf')
    svm_plot(clf.decision_function)

def svc_run():
    params = svc_optimal_params()
    clf = svc_fit(params)
    svm_plot(clf.decision_function)

if __name__ == '__main__':
    # svc_run()
    svm_run()
