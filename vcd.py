from sklearn.metrics import average_precision_score
from sklearn.externals import joblib
from sklearn.preprocessing import balance_weights

import numpy as np
import time
import os
from sys import stdout
import util
from util import data
import logging

STDOUT_HANDLER = logging.StreamHandler(stdout)

class VisualConceptDetection(object):
    """
    Contains methods to train and evaluate
    classifiers for the Visual Concept Detection task.

    Args:
        classifier: A scikit-learn Classifier.
        datamanager: A DataManager instance used to load
            training and test data.
        log_file (optional): Path to a file, where the output should be logged.
            Additionally log messages are always printed to stdout.
    """
    def __init__(self, classifier, datamanager, log_file = None):
        self.classifier = classifier
        self.init_logger(log_file)
        self.datamanager = datamanager

    def init_logger(self, log_file = None):
        self.logger = logging.getLogger(str(log_file))
        self.logger.setLevel("INFO")
        self.logger.addHandler(STDOUT_HANDLER)

        if log_file:
            dirname = os.path.dirname(log_file)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            self.logger.addHandler(logging.FileHandler(log_file, mode='w'))

    def evaluate_training_set(self, category):
        X = self.datamanager.build_sample_matrix("train", category)
        Y = self.datamanager.build_class_vector("train", category)
        return self.evaluate(X, Y)

    def evaluate_test_set(self, category):
        X = self.datamanager.build_sample_matrix("test", category)
        Y = self.datamanager.build_class_vector("test", category)
        return self.evaluate(X, Y)

    def evaluate(self, X_test, y_test):
        """Evaluate the classification performance of self.classifier
        on X_test and y_test.

        If you just want to evaluate performance on the training or
        test set, use evaluate_training_set and evaluate_test_set
        instead.

        Returns accuracy and average precision.
        """
        self.logger.info("Compute accuracy: ")
        self.logger.info("sample_length: %d" % X_test.shape[0])
        t2 = time.time()
        accuracy = self.classifier.score(X_test, y_test)
        self.logger.info(accuracy)
        self.logger.info("%f seconds\n" % (time.time() - t2))

        self.logger.info("Compute average precision: ",)
        t2 = time.time()
        if hasattr(self.classifier, "decision_function"):
            predictions = self.classifier.decision_function(X_test)
        else: # Some classifiers don't implement decision_function(), (e.g. RandomForest)
            predictions = self.classifier.predict_proba(X_test)[:,1]

        avg_precision = average_precision_score(y_test, predictions)
        self.logger.info(avg_precision)
        self.logger.info("%f seconds\n" % (time.time() - t2))

        return (accuracy, avg_precision)

    def fit_category(self, category, weighted=False):
        """Fit self.classifier to category.

        Args:
            category: A string specifying the category to learn.
            weighted: Whether to weight the training samples
                according to the class distribution in the sample data.
                Default is False.
        """
        X_train = self.datamanager.build_sample_matrix("train", category)

        self.logger.info("Training category %s", category)

        self.logger.info("Building training class vector")
        t2 = time.time()
        y_train = self.datamanager.build_class_vector("train", category)
        self.logger.info("%f seconds\n" % (time.time() - t2))

        if weighted:
            weights = self.calculate_weights(y_train)
            self.logger.info(weights)
        else:
            weights = np.ones_like(y_train, dtype=np.float32)

        self.logger.info("Fitting classifier to data")
        stdout.flush()
        t2 = time.time()
        print "X: ", X_train
        print "y: ", y_train
        try:
            self.classifier.fit(X_train, y_train, sample_weight=weights)
        except TypeError:
            self.logger.warning("classifier's fit method doesn't support sample weights")
            self.classifier.fit(X_train, y_train)
            self.logger.info("%f seconds\n" % (time.time() - t2))

    def calculate_weights(self, classes):
        return balance_weights(classes)

    def count_positives(self, classes):
        """Count the number of positive examples in the data."""
        return np.count_nonzero(classes)

    def count_negatives(self, classes):
        """Count the number of negative examples in the data."""
        return len(classes) - self.count_positives(classes)

    def dump_object(self, obj, fname, category="", classifier=None):
        """Serialize an object to disk.

        This is most often used to serialize trained classifiers,
        so they can be reloaded for later processing (e.g. visualization).
        The object is serialized into a file in a subfolder
        of the classifier-folder of self.datamanager.
        The path will mirror the parameter settings of
        the classifier object.
        Objects that were serialized with this method can
        be re-loaded with load_object.

        Args:
            obj: A python object, that should be serialized (e.g. a classifier).
            fname: Name of the serialized file.
            category: String that specifies, for which category the
                classifier was trained. This introduces another directory level
                to separate classifiers for different categories.
            classifier: Classifier object, that determines the path
                of the serialized file. The directory structure
                mirrors the parameters of this classifier.
                By default, self.classifier is used.
        """
        self.logger.info("Writing object to disk")
        t2 = time.time()
        if classifier == None:
            classifier = self.classifier

        try:
            folder = util.folder_name(self.datamanager.PATHS["CLASSIFIER"], category, classifier)
            if not os.path.isdir(folder):
                os.makedirs(folder)

            joblib.dump(obj, os.path.join(folder, fname), compress=3)
        except Exception as e:
            self.logger.error("Joblib failed: %s" % e)
        self.logger.info("%f seconds\n" % (time.time() - t2))

    def load_object(self, fname, category="", classifier=None):
        """Load an object, that was serialized with dump_object.

        Args:
            fname: Name of the serialized file.
            category: Category name, that was used for serialization.
            classifier: Classifier object, that was used for serialization.
        """
        self.logger.info("Reading object from disk")
        t2 = time.time()
        if classifier == None:
            classifier = self.classifier

        try:
            folder = util.folder_name(self.datamanager.PATHS["CLASSIFIER"], category, classifier)
            if not os.path.isdir(folder):
                self.logger.info("Object's path doesn't exist")
                return None

            obj = joblib.load(os.path.join(folder, fname))
            self.logger.info("%f seconds\n" % (time.time() - t2))
            return obj
        except Exception as e:
            self.logger.error("Joblib failed: %s" % e)
            return None

    def run(self, category):
        """Do a complete classification run for category.

        Trains self.classifier on category, dumps the resulting
        classifier and evaluates it on the training and test set.
        """
        self.fit_category(category, False)
        self.dump_object(self.classifier, "Classifier", category)

        self.logger.info("Evaluate on training set")
        self.evaluate_training_set(category)
        self.logger.info("Evaluate on test set")
        self.evaluate_test_set(category)

