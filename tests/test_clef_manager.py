import unittest
import os
import numpy as np
from sklearn.decomposition import PCA
from datamanagers.ClefManager import CLEFManager
from datamanagers import InvalidDatasetException, NoSuchCategoryException
from MyTestDataManager import *

class TestClefManager(unittest.TestCase):
    def setUp(self):
        self.datamanager = CLEFManager()
        self.datamanager.change_base_path(os.path.join(BASE_PATH, "testdata"))

    def test_invalid_dataset_clef(self):
        self.assertRaises(InvalidDatasetException, self.datamanager.build_sample_matrix, "rubbish", "test")

    def test_invalid_dataset_clef2(self):
        self.assertRaises(InvalidDatasetException, self.datamanager.build_class_vector, "rubbish", "test")

    def test_invalid_category_clef(self):
        self.assertRaises(NoSuchCategoryException, self.datamanager.get_positive_samples, "test", "rubbish")

    def test_sample_matrix_pca_clef(self):
        dm = MyTestDataManager()
        dm.use_pca(n_components = 1)

        samples = dm.build_sample_matrix("all")
        should_be = np.array([
            [0.191152],
            [-0.42905428],
            [0.26799257],
            [0.80695484],
            [-0.83704513],
            [-0.27844463],
            [0.67095735],
            [0.06459591],
            [-0.03133069]], dtype=np.float32)
        difference_matrix = np.abs(samples - should_be)
        self.assertTrue((difference_matrix < 0.000001).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, samples))

    def test_training_sample_matrix(self):
        samples = self.datamanager.build_sample_matrix("train")
        should_be = np.array([
            [ 0.12442154,  0.57743013,  0.9548108 ,  0.22592719,  0.10155164, 0.60750473],
            [ 0.53320956,  0.18181397,  0.60112703,  0.09004746,  0.31448245, 0.85619318],
            [ 0.44842428,  0.50402522,  0.45302102,  0.54796243,  0.82176286, 0.11623112],
            [ 0.18139255,  0.83218205,  0.87969971,  0.81630158,  0.57571691, 0.08127511],
            [ 0.31588301,  0.05166245,  0.16203263,  0.02196996,  0.96935761, 0.9854272 ]
            ], dtype=np.float32)
        difference_matrix = np.abs(samples - should_be)
        self.assertTrue((difference_matrix < 0.00000001).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, samples))

    def test_test_sample_matrix(self):
        samples = self.datamanager.build_sample_matrix("test")
        should_be = np.array([
            [ 0.64663881,  0.55629711,  0.11966438,  0.04559849,  0.69156636, 0.4500224 ],
            [ 0.08660618,  0.83642531,  0.9239062 ,  0.53778457,  0.56708116, 0.13766008],
            [ 0.31313366,  0.88874122,  0.20000355,  0.56186443,  0.15771926, 0.81349361],
            [ 0.38948518,  0.33885501,  0.567841  ,  0.36167425,  0.18220702, 0.57701336]
            ], dtype=np.float32)
        difference_matrix = np.abs(samples - should_be)
        self.assertTrue((difference_matrix < 0.00000001).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, samples))

    def test_complete_sample_matrix(self):
        samples = self.datamanager.build_sample_matrix("all")
        should_be = np.array([
            [ 0.12442154,  0.57743013,  0.9548108 ,  0.22592719,  0.10155164, 0.60750473],
            [ 0.53320956,  0.18181397,  0.60112703,  0.09004746,  0.31448245, 0.85619318],
            [ 0.44842428,  0.50402522,  0.45302102,  0.54796243,  0.82176286, 0.11623112],
            [ 0.18139255,  0.83218205,  0.87969971,  0.81630158,  0.57571691, 0.08127511],
            [ 0.31588301,  0.05166245,  0.16203263,  0.02196996,  0.96935761, 0.9854272 ],
            [ 0.64663881,  0.55629711,  0.11966438,  0.04559849,  0.69156636, 0.4500224 ],
            [ 0.08660618,  0.83642531,  0.9239062 ,  0.53778457,  0.56708116, 0.13766008],
            [ 0.31313366,  0.88874122,  0.20000355,  0.56186443,  0.15771926, 0.81349361],
            [ 0.38948518,  0.33885501,  0.567841  ,  0.36167425,  0.18220702, 0.57701336]
            ], dtype=np.float32)
        difference_matrix = np.abs(samples - should_be)
        self.assertTrue((difference_matrix < 0.00000001).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, samples))

    def test_training_class_vector(self):
        classes = self.datamanager.build_class_vector("train", "test")
        should_be = np.array([0, 0, 1, 0, 1])
        self.assertTrue((classes==should_be).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, classes))

    def test_test_class_vector(self):
        classes = self.datamanager.build_class_vector("test", "test")
        should_be = np.array([1, 0, 0, 1])
        self.assertTrue((classes==should_be).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, classes))

    def test_complete_class_vector(self):
        classes = self.datamanager.build_class_vector("all", "test")
        should_be = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1])
        self.assertTrue((classes==should_be).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, classes))
