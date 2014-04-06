import unittest
from datamanagers.CaltechManager import CaltechManager
from datamanagers import InvalidDatasetException, NoSuchCategoryException
import os
import numpy as np
from sklearn.decomposition import PCA
from test_datamanager import BASE_PATH

class TestCaltechManager(unittest.TestCase):
    def setUp(self):
        self.datamanager = CaltechManager()
        self.datamanager.change_base_path(os.path.join(BASE_PATH, "testdata"))

    def test_invalid_dataset_caltech(self):
        self.assertRaises(InvalidDatasetException, self.datamanager.build_sample_matrix, "rubbish", "test")

    def test_invalid_dataset_caltech2(self):
        self.assertRaises(InvalidDatasetException, self.datamanager.build_class_vector, "rubbish", "test")

    def test_invalid_category_caltech(self):
        self.assertRaises(NoSuchCategoryException, self.datamanager.get_positive_samples, "test", "rubbish")

    def test_invalid_category_caltech2(self):
        self.assertRaises(NoSuchCategoryException, self.datamanager.build_sample_matrix, "test", "rubbish")

    def test_training_sample_matrix(self):
        samples = self.datamanager.build_sample_matrix("train", "TestFake")
        should_be = np.array([
            [ 0.44842428,  0.50402522,  0.45302102,  0.54796243,  0.82176286, 0.11623112],
            [ 0.31588301,  0.05166245,  0.16203263,  0.02196996,  0.96935761, 0.9854272 ],
            [ 0.12442154,  0.57743013,  0.9548108 ,  0.22592719,  0.10155164, 0.60750473],
            [ 0.53320956,  0.18181397,  0.60112703,  0.09004746,  0.31448245, 0.85619318],
            [ 0.18139255,  0.83218205,  0.87969971,  0.81630158,  0.57571691, 0.08127511]
            ], dtype=np.float32)
        difference_matrix = np.abs(samples - should_be)
        self.assertTrue((difference_matrix < 0.00000001).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, samples))

    def test_test_sample_matrix(self):
        samples = self.datamanager.build_sample_matrix("test", "TestFake")
        should_be = np.array([
            [ 0.64663881,  0.55629711,  0.11966438,  0.04559849,  0.69156636, 0.4500224 ],
            [ 0.38948518,  0.33885501,  0.567841  ,  0.36167425,  0.18220702, 0.57701336],
            [ 0.08660618,  0.83642531,  0.9239062 ,  0.53778457,  0.56708116, 0.13766008],
            [ 0.31313366,  0.88874122,  0.20000355,  0.56186443,  0.15771926, 0.81349361]
            ], dtype=np.float32)
        difference_matrix = np.abs(samples - should_be)
        self.assertTrue((difference_matrix < 0.00000001).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, samples))

    def test_all_sample_matrix(self):
        samples = self.datamanager.build_sample_matrix("all", "TestFake")
        should_be = np.array([
            [ 0.44842428,  0.50402522,  0.45302102,  0.54796243,  0.82176286, 0.11623112],
            [ 0.31588301,  0.05166245,  0.16203263,  0.02196996,  0.96935761, 0.9854272 ],
            [ 0.64663881,  0.55629711,  0.11966438,  0.04559849,  0.69156636, 0.4500224 ],
            [ 0.38948518,  0.33885501,  0.567841  ,  0.36167425,  0.18220702, 0.57701336],
            [ 0.12442154,  0.57743013,  0.9548108 ,  0.22592719,  0.10155164, 0.60750473],
            [ 0.53320956,  0.18181397,  0.60112703,  0.09004746,  0.31448245, 0.85619318],
            [ 0.18139255,  0.83218205,  0.87969971,  0.81630158,  0.57571691, 0.08127511],
            [ 0.08660618,  0.83642531,  0.9239062 ,  0.53778457,  0.56708116, 0.13766008],
            [ 0.31313366,  0.88874122,  0.20000355,  0.56186443,  0.15771926, 0.81349361]
            ], dtype=np.float32)
        difference_matrix = np.abs(samples - should_be)
        self.assertTrue((difference_matrix < 0.00000001).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, samples))

    def test_all_sample_matrix_exclude_feature(self):
        self.datamanager.exclude_feature = 4
        samples = self.datamanager.build_sample_matrix("all", "TestFake")
        should_be = np.array([
            [ 0.44842428,  0.50402522,  0.45302102,  0.54796243,  0.11623112],
            [ 0.31588301,  0.05166245,  0.16203263,  0.02196996,  0.9854272 ],
            [ 0.64663881,  0.55629711,  0.11966438,  0.04559849,  0.4500224 ],
            [ 0.38948518,  0.33885501,  0.567841  ,  0.36167425,  0.57701336],
            [ 0.12442154,  0.57743013,  0.9548108 ,  0.22592719,  0.60750473],
            [ 0.53320956,  0.18181397,  0.60112703,  0.09004746,  0.85619318],
            [ 0.18139255,  0.83218205,  0.87969971,  0.81630158,  0.08127511],
            [ 0.08660618,  0.83642531,  0.9239062 ,  0.53778457,  0.13766008],
            [ 0.31313366,  0.88874122,  0.20000355,  0.56186443,  0.81349361]
            ], dtype=np.float32)
        difference_matrix = np.abs(samples - should_be)
        self.assertTrue((difference_matrix < 0.00000001).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, samples))

    @unittest.expectedFailure # TODO: dependent on file order
    def test_complete_sample_matrix(self):
        samples = self.datamanager.build_complete_sample_matrix("train")
        should_be = np.array([
            [ 0.31313366,  0.88874122,  0.20000355,  0.56186443,  0.15771926, 0.81349361],
            [ 0.12442154,  0.57743013,  0.9548108 ,  0.22592719,  0.10155164, 0.60750473],
            [ 0.53320956,  0.18181397,  0.60112703,  0.09004746,  0.31448245, 0.85619318],
            [ 0.18139255,  0.83218205,  0.87969971,  0.81630158,  0.57571691, 0.08127511],
            [ 0.44842428,  0.50402522,  0.45302102,  0.54796243,  0.82176286, 0.11623112],
            [ 0.31588301,  0.05166245,  0.16203263,  0.02196996,  0.96935761, 0.9854272 ],
            ], dtype=np.float32)
        difference_matrix = np.abs(samples - should_be)
        self.assertTrue((difference_matrix < 0.00000001).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, samples))

    @unittest.expectedFailure # TODO: dependent on file order
    def test_complete_sample_matrix_exclude_feature(self):
        self.datamanager.exclude_feature = 1
        samples = self.datamanager.build_complete_sample_matrix("train")
        should_be = np.array([
            [ 0.31313366,  0.20000355,  0.56186443,  0.15771926, 0.81349361],
            [ 0.12442154,  0.9548108 ,  0.22592719,  0.10155164, 0.60750473],
            [ 0.53320956,  0.60112703,  0.09004746,  0.31448245, 0.85619318],
            [ 0.18139255,  0.87969971,  0.81630158,  0.57571691, 0.08127511],
            [ 0.44842428,  0.45302102,  0.54796243,  0.82176286, 0.11623112],
            [ 0.31588301,  0.16203263,  0.02196996,  0.96935761, 0.9854272 ],
            ], dtype=np.float32)
        difference_matrix = np.abs(samples - should_be)
        self.assertTrue((difference_matrix < 0.00000001).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, samples))

    def test_complete_sample_matrix_fail(self):
        self.assertRaises(NotImplementedError, self.datamanager.build_complete_sample_matrix, "all")

    def test_training_class_vector(self):
        classes = self.datamanager.build_class_vector("train", "TestFake")
        should_be = np.array([1, 1, 0, 0, 0])
        self.assertTrue((classes==should_be).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, classes))

    def test_test_class_vector(self):
        classes = self.datamanager.build_class_vector("test", "TestFake")
        should_be = np.array([1, 1, 0, 0])
        self.assertTrue((classes==should_be).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, classes))

    def test_complete_class_vector(self):
        classes = self.datamanager.build_class_vector("all", "TestFake")
        should_be = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.assertTrue((classes==should_be).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, classes))

    def test_sample_matrix_pca(self):
        self.datamanager.use_pca(n_components = 1)

        samples = self.datamanager.build_sample_matrix("all", "TestFake")
        should_be = np.array([
            [-0.24263228],
            [0.85717554],
            [0.29054203],
            [0.03857126],
            [-0.18379566],
            [0.44021899],
            [-0.78841356],
            [-0.65111911],
            [-0.08255303]
        ], dtype=np.float32)

        difference_matrix = np.abs(samples - should_be)
        self.assertTrue((difference_matrix < 0.000001).all(), "Should be:\n%s\nbut is:\n%s" % (should_be, samples))