"""
This class is derived from GridSearchCV and overwrites the _fit method
to allow weighing the samples for grid search.

This version of _fit is identical to the original one,
except for the parallel call to the fit_grid_point function,
which was modified to pass the weights of the current training samples.
"""
from sklearn.utils.validation import _num_samples, check_arrays
from sklearn.metrics.scorer import _deprecate_loss_and_score_funcs
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.cross_validation import check_cv
from sklearn.grid_search import fit_grid_point, _CVScoreTuple
from sklearn.grid_search import GridSearchCV
from sklearn.base import is_classifier, clone
from sklearn.preprocessing import balance_weights
import warnings
import numpy as np

class WeightedGridSearchCV(GridSearchCV):
    def _fit(self, X, y, parameter_iterable):
        """Actual fitting, performing the search over parameters."""

        estimator = self.estimator
        cv = self.cv

        n_samples = _num_samples(X)
        X, y = check_arrays(X, y, allow_lists=True, sparse_format='csr')

        self.scorer_ = _deprecate_loss_and_score_funcs(
            self.loss_func, self.score_func, self.scoring)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
            y = np.asarray(y)
        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

        if self.verbose > 0:
            if isinstance(parameter_iterable, Sized):
                n_candidates = len(parameter_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(len(cv), n_candidates,
                                         n_candidates * len(cv)))

        base_estimator = clone(self.estimator)

        pre_dispatch = self.pre_dispatch

        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch)(
                delayed(fit_grid_point)(
                    X, y, base_estimator, parameters, train, test, self.scorer_,
                    self.verbose, **{'sample_weight': balance_weights(y[train])}) for parameters in
                parameter_iterable for train, test in cv)

        # Out is a list of triplet: score, estimator, n_test_samples
        n_fits = len(out)
        n_folds = len(cv)

        scores = list()
        grid_scores = list()
        for grid_start in range(0, n_fits, n_folds):
            n_test_samples = 0
            score = 0
            all_scores = []
            for this_score, parameters, this_n_test_samples in \
                    out[grid_start:grid_start + n_folds]:
                all_scores.append(this_score)
                if self.iid:
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                score += this_score
            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_folds)
            scores.append((score, parameters))
            # TODO: shall we also store the test_fold_sizes?
            grid_scores.append(_CVScoreTuple(
                parameters,
                score,
                np.array(all_scores)))
        # Store the computed scores
        self.grid_scores_ = grid_scores

        # Find the best parameters by comparing on the mean validation score:
        # note that `sorted` is deterministic in the way it breaks ties
        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best.parameters)
            if y is not None:
                best_estimator.fit(X, y, sample_weight = balance_weights(y),**self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self
