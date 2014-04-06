import numpy as np
import unittest
from util.data import storage as ds
import os
from MyTestDataManager import BASE_PATH

class TestStorage(unittest.TestCase):
    def setUp(self):
        self.test_matrix = np.array([
                        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                        [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16],
                        [0.03, 0.00, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24],
                        [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32],
                        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
                        [0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48]], np.float32)

        self.test_matrix_uint8 = np.array([
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [2, 4, 6, 8, 10, 12, 14, 16],
                        [3, 6, 9, 12, 15, 18, 21, 24],
                        [4, 8, 12, 16, 20, 24, 28, 32],
                        [5, 10, 15, 20, 25, 30, 35, 40],
                        [6, 12, 18, 24, 30, 36, 42, 48]], np.uint8)
        
        self.test_matrix_uint32 = np.array([
                        [10023, 20023, 30023, 40023, 50023, 60023, 70023, 812932],
                        [20023, 40023, 60023, 80023, 100023, 120023, 140023, 1612932],
                        [30023, 60023, 90023, 120023, 150023, 180023, 210023, 2412932],
                        [40023, 80023, 120023, 160023, 200023, 240023, 280023, 3212932],
                        [50023, 100023, 150023, 200023, 250023, 300023, 350023, 4012932],
                        [60023, 120023, 180023, 240023, 300023, 360023, 420023, 4812354]], np.uint32)
        
        # the image used to generate the test keypoints file
        # had this size:
        image_size = (400, 225)
        self.test_keypoints = [ds.Keypoint(x, y) for x in range(0, image_size[0], 6) for y in range(0, image_size[1], 6)]
    
    def test_load_float_matrix(self):
        loaded_matrix = ds.load_matrix(os.path.join(BASE_PATH, "testdata", "test.mat"))
        self.assertTrue((self.test_matrix == loaded_matrix).all(), "Should be:\n%s\nbut is:\n%s" % (self.test_matrix, loaded_matrix))
        
    def test_save_float_matrix(self):
        ds.save_matrix(self.test_matrix, os.path.join(BASE_PATH, "testdata", "writetest.mat"))
        loaded = ds.load_matrix(os.path.join(BASE_PATH, "testdata", "writetest.mat"))
        self.assertTrue((self.test_matrix == loaded).all(), "Should be:\n%s\nbut is:\n%s" % (self.test_matrix, loaded))
        
    def test_save_uint8_matrix(self):
        ds.save_matrix(self.test_matrix_uint8, os.path.join(BASE_PATH, "testdata", "writetest_uint8.mat"))
        loaded = ds.load_matrix(os.path.join(BASE_PATH, "testdata", "writetest_uint8.mat"))
        self.assertTrue((self.test_matrix_uint8 == loaded).all(), "Should be:\n%s\nbut is:\n%s" % (self.test_matrix_uint8, loaded))
    
    def test_save_uint32_matrix(self):
        ds.save_matrix(self.test_matrix_uint32, os.path.join(BASE_PATH, "testdata", "writetest_uint32.mat"))
        loaded = ds.load_matrix(os.path.join(BASE_PATH, "testdata", "writetest_uint32.mat"))
        self.assertTrue((self.test_matrix_uint32 == loaded).all(), "Should be:\n%s\nbut is:\n%s" % (self.test_matrix_uint32, loaded))
        
    def test_matrices_binary_equal(self):
        pass
        
    def test_load_keypoints(self):
        loaded = ds.keypoints_from_file(os.path.join(BASE_PATH, "testdata", "features", "test.kps"))
        self.assertEqual(len(loaded), len(self.test_keypoints))
        for k1, k2 in zip(loaded, self.test_keypoints):
            self.assertEqual(k1, k2)
            
    def test_save_keypoints(self):
        ds.keypoints_to_file(self.test_keypoints, os.path.join(BASE_PATH, "testdata", "features", "writetest.kps"))
        loaded = ds.keypoints_from_file(os.path.join(BASE_PATH, "testdata", "features", "writetest.kps"))
        self.assertEqual(len(loaded), len(self.test_keypoints))
        for k1, k2 in zip(loaded, self.test_keypoints):
            self.assertEqual(k1, k2)

    def test_keypoints_not_equal(self):
        kp1 = self.test_keypoints[0]
        kp2 = self.test_keypoints[1]
        self.assertTrue(kp1 != kp2)
        self.assertFalse(kp1 != kp1)

    def test_uncompressed_matrix(self):
        self.assertRaises(NotImplementedError, ds.load_matrix, os.path.join(BASE_PATH, "testdata", "uncompressed_test.mat"))

    def test_unsupported_matrix_format(self):
        self.assertRaises(NotImplementedError, ds.save_matrix, np.array([], dtype=np.int64), "fail.mat")

    def test_unsupported_matrix_format_load(self):
        self.assertRaises(NotImplementedError, ds.load_matrix, os.path.join(BASE_PATH, "testdata", "wrong_mtype.mat"))