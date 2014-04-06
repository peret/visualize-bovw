import unittest
from datamanagers import DataManager, InvalidDatasetException
from MyTestDataManager import *
import numpy as np


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

class TestDataManager(unittest.TestCase):

    def setUp(self):
        self.datamanager = DataManager()

    def test_invalid_dataset(self):
        self.assertRaises(InvalidDatasetException, self.datamanager.get_positive_samples, "rubbish", "test")

    def test_invalid_dataset2(self):
        self.assertRaises(InvalidDatasetException, self.datamanager.build_sample_matrix, "rubbish", "test")

    def test_methods_not_implemented(self):
        self.assertRaises(NotImplementedError, self.datamanager.get_positive_samples, "all", "test")
        self.assertRaises(NotImplementedError, self.datamanager._build_sample_matrix, "all", "test")
        self.assertRaises(NotImplementedError, self.datamanager._build_class_vector, "all", "test")