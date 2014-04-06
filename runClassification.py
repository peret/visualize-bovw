"""Helper script to run the training for a single category."""

import sys
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from vcd import VisualConceptDetection
import os
import time
from util import svm
from datamanagers.CaltechManager import CaltechManager
import numpy as np
import pylab as pl

from runGridSearch import GridSearch

if __name__ == "__main__":
    category = "airplanes"
    total = time.time()

    clf = RandomForestClassifier(n_estimators = 2000)

    # clf = AdaBoostClassifier(n_estimators = 2000)
    # clf.base_estimator.max_depth = 4

    # clf = LinearSVC(C=100)
    # clf = SVC(C=10)

    dm = CaltechManager()
    vcd = VisualConceptDetection(classifier=clf, datamanager=dm)

    vcd.run(category)
    print "Total execution time: %f minutes" % ((time.time() - total) / 60.0)
