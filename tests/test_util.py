import unittest
import util
import os
from sklearn.ensemble import AdaBoostClassifier

class TestUtil(unittest.TestCase):
    def test_bow_descriptor_name(self):
        self.assertEqual(util.bow_name("/test/test2/{1234}.jpg"), "/test/test2/{1234}.descr_bowdescr.bin")
        self.assertEqual(util.bow_name("{1234}.jpg"), "{1234}.descr_bowdescr.bin")
        self.assertEqual(util.bow_name("{1234}"), "{1234}.descr_bowdescr.bin")

    def test_descriptors_name(self):
        self.assertEqual(util.descriptors_name("/test/test2/{1234}.jpg"), "/test/test2/{1234}.descr.mat")
        self.assertEqual(util.descriptors_name("{1234}.jpg"), "{1234}.descr.mat")
        self.assertEqual(util.descriptors_name("{1234}"), "{1234}.descr.mat")

    def test_params_to_path(self):
        clf = AdaBoostClassifier()
        params = {"first": 12, "third": 23.42, "second": "hello", "base_estimator": clf}
        self.assertEqual(util.params_to_path(params), "base_estimator__AdaBoostClassifier/first__12/second__hello/third__23.42")

    def test_folder_name(self):
        clf = AdaBoostClassifier(n_estimators=23)
        clf.base_estimator.max_depth = 42
        base = "/hello/world/"
        category = "testing"
        params_path = util.params_to_path(clf.get_params())

        self.assertEqual(
            util.folder_name(base, category, clf),
            os.path.join("/hello/world/AdaBoostClassifier/testing/", params_path))
