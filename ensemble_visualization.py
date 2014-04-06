from visualization import Visualization
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from vcd import VisualConceptDetection
from itertools import izip
from datamanagers.CaltechManager import CaltechManager

import os
import numpy as np

class EnsembleVisualization(Visualization):
    def get_image_title(self, prediction, real):
        """Returns a string that describes whether the prediction
        is a true positive, false positive, etc. and with what
        confidence the prediction is made.

        Args:
            prediction: List of predicted probabilities of
            the respective classes.
            real: List of corresponding correct labels.
        """
        p = np.argmax(prediction)
        result = ""
        result += "True " if p == real else "False "
        result += "positive" if p == 1 else "negative"
        result += " - confidence: %.5f" % prediction[p]
        return result

if __name__ == "__main__":
    # ada = AdaBoostClassifier()
    # ada.n_estimators = 50
    # ada.base_estimator.max_depth = 1

    random_forest = RandomForestClassifier(n_estimators=100)

    category = "trilobite"
    dataset = "all"
    datamanager = CaltechManager()
    datamanager.PATHS["RESULTS"] = os.path.join(datamanager.PATHS["BASE"], "results_trilobite_rf_testing")

    # vcd = VisualConceptDetection(ada, datamanager)
    vcd = VisualConceptDetection(random_forest, datamanager)

    clf = vcd.load_object("Classifier", category)
    feature_importances = clf.feature_importances_

    sample_matrix = vcd.datamanager.build_sample_matrix(dataset, category)
    class_vector = vcd.datamanager.build_class_vector(dataset, category)
    pred = clf.predict_proba(sample_matrix)

    vis = EnsembleVisualization(datamanager)
    del clf
    image_titles = [vis.get_image_title(prediction, real) for prediction, real in
                    izip(pred, class_vector)]
    del class_vector
    del sample_matrix

    img_names = [f for f in vcd.datamanager.get_image_names(dataset, category)]
    vis.visualize_images(img_names, feature_importances, image_titles)
