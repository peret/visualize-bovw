import os
import time
import numpy as np

from vcd import VisualConceptDetection
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, train_test_split

from weighted_grid_search import WeightedGridSearchCV
from util import class_name
from util import data
from datetime import datetime


class GridSearch(VisualConceptDetection):
    """Provides methods to perform a grid search
    for a specific category and classifier.
    """

    def __init__(self, classifier, datamanager, category):
        self.now = str(datetime.now())
        self.log_path = os.path.join(datamanager.PATHS["LOGS"],
                    "grid_search",
                    class_name(classifier),
                    category)
        log_file = os.path.join(self.log_path, self.now + ".log")

        super(GridSearch, self).__init__(classifier, datamanager, log_file = log_file)
        self.category = category
        self.grid_search_obj = None

    def grid_search(self, params, X=None, y=None, weight_samples=False):
        """Perform a nested grid search.

        Args:
            params: Parameter hash, specifiying the parameter configurations
                for the grid search (see http://scikit-learn.org/stable/modules/grid_search.html).
            X: Matrix of sample data.
            y: Vector of class labels.
            weight_samples: Whether to use grid search with weighted
                samples or not.
        """
        self.logger.info("Grid search for %s at %s" %
            (class_name(self.classifier), datetime.now()))
        self.logger.info("Category: %s" % self.category)
        self.logger.info("Parameters: %s" % params)
        self.logger.info("Initial sample weighting: %s" % weight_samples)
        result = self.search(params, X, y, weight_samples)
        joblib.dump(result, os.path.join(self.log_path, self.now + ".grid"), compress=3)
        return result

    def search(self, params, X, y, weight_samples):
        """Perform the actual nested grid search.

        grid_search should be called instead.
        """
        if X is None:
            X = self.datamanager.build_sample_matrix("train", self.category)
        if y is None:
            y = self.datamanager.build_class_vector("train", self.category)

        if weight_samples:
            GridSearchClass = WeightedGridSearchCV
        else:
            GridSearchClass = GridSearchCV

        outer_kfold = StratifiedKFold(y, n_folds=4)
        self.logger.info("Starting Nested Grid Search")
        grid_searches = []
        results = []

        for train_index, test_index in outer_kfold:
            cross_validation = StratifiedKFold(y[train_index], n_folds=3)
            grid_searches.append(GridSearchClass(self.classifier, params, refit=True, scoring="average_precision",
                                   n_jobs=-1, cv=cross_validation))

            t1 = time.time()
            grid_searches[-1].fit(X[train_index], y[train_index])
            self.logger.info("%f minutes" % ((time.time() - t1) / 60.0))
            self.logger.info(grid_searches[-1].best_score_)
            self.logger.info(grid_searches[-1].best_params_)
            self.classifier = grid_searches[-1].best_estimator_
            acc, avp = self.evaluate(X[test_index], y[test_index])
            results.append(avp)

        self.grid_search_obj = grid_searches[np.argmax(np.array(results))]

        # refit the best estimator with complete dataset
        self.grid_search_obj.best_estimator_.fit(X, y)
        return self.grid_search_obj
