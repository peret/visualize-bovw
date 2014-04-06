import os
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics.pairwise import additive_chi2_kernel, chi2_kernel
from util import data
from util import generate_evaluation_summary
from datamanagers.CaltechManager import CaltechManager
import numpy as np
import warnings
from grid_search import GridSearch


def approximate_gamma(sample_matrix):
    """ Approximates the width parameter for the gaussian kernel.

        By computing the average distance between all training samples,
        we can approximate the width parameter of the gaussian and eliminate
        the need to optimize it through grid search.
    """
    return np.mean(-additive_chi2_kernel(sample_matrix))

def build_train_kernels(categories, datamanager):
    kernels = []
    gammas = []
    for c in categories:
        X = datamanager.build_sample_matrix("train", c)
        gamma = approximate_gamma(X)
        gammas.append(gamma)
        kernels.append(chi2_kernel(X, X, gamma=1.0/gamma))
    return kernels, gammas

if __name__ == "__main__":
    total = time.time()
    params = {"n_estimators": [10, 50, 100, 200, 400, 750, 800, 1000, 2000], "base_estimator__max_depth": [1, 2, 3, 5], "base_estimator__random_state": [0], "random_state": [0]}
#    params = {"C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

    datamanager = CaltechManager()
    categories = [c for c in os.listdir(datamanager.PATHS["CATEGORIES_DIR"]) if c != datamanager.BACKGROUND and os.path.splitext(c)[1] != ".py"]

    #kernels, gammas = build_train_kernels(categories, datamanager)
    #print "Finished building kernels"

    #grids = (GridSearch(SVC(kernel="precomputed"), c) for c in categories)
    # grids = (GridSearch(RandomForestClassifier(), c) for c in categories)

    grids = [GridSearch(AdaBoostClassifier(), datamanager, c) for c in categories]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g in grids:
            g.grid_search(params, weight_samples=False)
        generate_evaluation_summary(grids, "grid_test.csv")

    print "Total execution time: %f minutes" % ((time.time() - total) / 60.0)
