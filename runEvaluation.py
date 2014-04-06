from vcd import VisualConceptDetection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from util import class_name
from datetime import datetime
import os
from datamanagers.CaltechManager import CaltechManager

def evaluate(category, clf, datamanager, data=(None, None)):
    """Run evaluation of a classifier, for one category.

    If data isn't set explicitly, the test set is
    used by default.
    """
    log_file = os.path.join(datamanager.PATHS["LOGS"], "evaluation", class_name(clf), category)
    log_file = os.path.join(log_file, str(datetime.now()) + ".log")

    vcd = VisualConceptDetection(None, datamanager, log_file = log_file)
    clf = vcd.load_object("Classifier", category, clf)
    vcd.classifier = clf
    if (data[0] is None) or (data[1] is None):
        return vcd.evaluate_test_set(category)
    else:
        return vcd.evaluate(X_test=data[0], y_test=data[1])

if __name__ == '__main__':
    # classifier = RandomForestClassifier()

    classifier = AdaBoostClassifier()
    classifier.n_estimators = 2000
    classifier.base_estimator.max_depth = 4

    # classifier = LinearSVC(C=100)

    category = "airplanes"
    datamanager = CaltechManager()
    evaluate(category, classifier, datamanager)
