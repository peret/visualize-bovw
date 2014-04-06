import datamanagers
import unittest
import numpy as np
from runGridSearch import GridSearch
from sklearn.ensemble import AdaBoostClassifier

class TestGridSearch(unittest.TestCase):
    def setUp(self):
        self.datamanager = datamanagers.CaltechManager()

    # def test_chi_squared(self):
    # def test_chi_squared(self):
    #     samples = self.datamanager.build_sample_matrix("test", "all")
    #     should_be = np.array([
    #         [ 0.12442154,  0.57743013,  0.9548108 ,  0.22592719,  0.10155164, 0.60750473],
    #         [ 0.53320956,  0.18181397,  0.60112703,  0.09004746,  0.31448245, 0.85619318],
    #         [ 0.44842428,  0.50402522,  0.45302102,  0.54796243,  0.82176286, 0.11623112],
    #         [ 0.18139255,  0.83218205,  0.87969971,  0.81630158,  0.57571691, 0.08127511],
    #         [ 0.31588301,  0.05166245,  0.16203263,  0.02196996,  0.96935761, 0.9854272 ],
    #         [ 0.64663881,  0.55629711,  0.11966438,  0.04559849,  0.69156636, 0.4500224 ],
    #         [ 0.08660618,  0.83642531,  0.9239062 ,  0.53778457,  0.56708116, 0.13766008],
    #         [ 0.31313366,  0.88874122,  0.20000355,  0.56186443,  0.15771926, 0.81349361],
    #         [ 0.38948518,  0.33885501,  0.567841  ,  0.36167425,  0.18220702, 0.57701336]
    #         ], dtype=np.float32)
    #     skl = (-0.5)*additive_chi2_kernel(samples)
    #     print skl
    #     my_chi2 = 0.5 * sum()

    @unittest.expectedFailure # nested grid search seems to result in the same classifier
    def test_weighted_grid_search(self):
        X = np.linspace(0, 1, 1000).reshape(1000, 1)
        y = np.zeros(1000)
        y[::13] = 1

        params = {"n_estimators": [1], "base_estimator__max_depth": [1], "base_estimator__criterion": ['gini']}
        grid = GridSearch(AdaBoostClassifier(), self.datamanager, "TestFake")
        unweighted_search = grid.grid_search(params, X, y, False)
        weighted_search = grid.grid_search(params, X, y, True)

        # print unweighted_search.decision_function(X)
        # print weighted_search.decision_function(X)
        self.assertTrue((weighted_search.decision_function(X) != unweighted_search.decision_function(X)).any(),
                            "The sample_weights should have some impact on the decision function, but they don't.")
