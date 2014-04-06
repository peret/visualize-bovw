import unittest
import datamanagers
from MyTestDataManager import *
from vcd import VisualConceptDetection
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
        
class TestVisualConceptDetection(unittest.TestCase):
    
    def setUp(self):
        self.datamanager = MyTestDataManager()
        clf = AdaBoostClassifier(n_estimators=14)
        clf.base_estimator.max_depth = 10
        self.vcd = VisualConceptDetection(clf, self.datamanager)
        
    def test_base_dir(self):
        path = os.path.join(BASE_PATH, "testdata", "bow")
        self.assertEqual(self.datamanager.PATHS["BOW"], path)

    # def test_dump_object(self):
        

    def test_dump_and_load_object(self):
        self.vcd.dump_object(self.vcd.classifier, "test_classifier")
        # TODO: meh
        self.assertIsInstance(self.vcd.load_object("test_classifier"), self.vcd.classifier.__class__)

    def test_count_methods(self):
        classes = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0])
        self.assertEqual(self.vcd.count_positives(classes), 5)
        self.assertEqual(self.vcd.count_negatives(classes), 7)

    def test_sample_weights(self):
        vec = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        p = 1.0
        n = 3.0 / 7.0
        weights = np.array([p, n, n, n, p, p, n, n, n, n])
        calculated = self.vcd.calculate_weights(vec)
        self.assertTrue((calculated==weights).all(), "Should be:\n%s\nbut is:\n%s" % (weights, calculated))
