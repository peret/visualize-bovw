from visualization import Visualization
from sklearn.svm import LinearSVC
from vcd import VisualConceptDetection
import numpy as np
from datamanagers.CaltechManager import CaltechManager

from itertools import izip

import sys
import os


def get_image_title(prediction, real):
    """Returns a string that describes whether the prediction
    is a true positive, false positive, etc. and with what
    confidence the prediction is made.

    Args:
        prediction: List of predicted probabilities of
        the respective classes.
        real: List of corresponding correct labels.
    """
    p = 1 if prediction > 0 else 0
    result = ""
    result += "True " if p == real else "False "
    result += "positive" if p == 1 else "negative"
    result += " - distance: %.5f" % prediction
    return result

def get_svm_importances(coef):
    """Normalize the SVM weights."""
    factor = 1.0 / np.linalg.norm(coef)
    return (coef * factor).ravel()

if __name__ == "__main__":
    svm = LinearSVC(C=0.1)

    category = "Faces"
    dataset = "all"
    datamanager = CaltechManager()
    datamanager.PATHS["RESULTS"] = os.path.join(datamanager.PATHS["BASE"], "results_Faces_LinearSVC_normalized")
    vcd = VisualConceptDetection(svm, datamanager)

    clf = vcd.load_object("Classifier", category)
    importances = get_svm_importances(clf.coef_)

    sample_matrix = vcd.datamanager.build_sample_matrix(dataset, category)
    class_vector = vcd.datamanager.build_class_vector(dataset, category)
    pred = clf.decision_function(sample_matrix)

    del clf
    image_titles = [get_image_title(prediction, real) for prediction, real in
                    izip(pred, class_vector)]
    del class_vector
    del sample_matrix

    img_names = [f for f in vcd.datamanager.get_image_names(dataset, category)]
    vis = Visualization(datamanager)
    vis.visualize_images(img_names, importances, image_titles)
