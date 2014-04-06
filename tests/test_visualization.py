import unittest
import numpy as np
from ensemble_visualization import EnsembleVisualization
from MyTestDataManager import *
import os
from matplotlib.colors import LinearSegmentedColormap

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.visualization = EnsembleVisualization(MyTestDataManager())

    def test_heatmap_data(self):
        self.assertTrue(True)

    def test_get_image_title(self):
        self.assertEqual(self.visualization.get_image_title([0.4, 0.6], 1), "True positive - confidence: 0.60000")
        self.assertEqual(self.visualization.get_image_title([0.9765, 0.0235], 0), "True negative - confidence: 0.97650")
        self.assertEqual(self.visualization.get_image_title([0.499, 0.501], 0), "False positive - confidence: 0.50100")
        self.assertEqual(self.visualization.get_image_title([0.5, 0.5], 1), "False negative - confidence: 0.50000")

    def test_get_max_importance(self):
        importances = [0.01, 0.2502, 0.1982, 0.00001]
        self.assertEqual(self.visualization.get_max_importance(importances), 0.2502)

    def test_get_min_importance(self):
        importances = [0.01, 0.2502, 0.1982, 0.00001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3425, 0.0012, 0.031412, 0.4634547, 0.03612, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(self.visualization.get_min_importance(importances), 0.03612)

    def test_get_min_importance_peak_at_last_index(self):
        importances = [0.01, 0.2502, 0.1982, 0.00001, 0.4634547, 0.4634547, 0.4634547, 0.4634547,
            0.4634547, 0.4634547, 0.3425, 0.0012, 0.031412, 0.4634547, 0.03612, 0.4634547, 0.4634547, 0.4634547]
        self.assertAlmostEqual(self.visualization.get_min_importance(importances), 0.4634547)

    def test_reverse_color_map(self):
        colors = {'blue': ((0.0, 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0)),
                'green': ((0.0, 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
                'red': ((0.0, 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5))}

        cmap = LinearSegmentedColormap('testcolormap', colors)
        reversed = {'blue': ((0.0, 0.0, 0.0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)),
                    'green': ((0.0, 0, 0), (0.09, 0, 0), (0.36, 1, 1), (0.625, 1, 1), (0.875, 0, 0), (1, 0, 0)),
                    'red': ((0.0, 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0))}
        result = self.visualization.reverse_color_map(cmap)
        for c in ["red", "green", "blue"]:
            for i, t in enumerate(reversed[c]):
                for j in range(3):
                    self.assertAlmostEqual(reversed[c][i][j], result._segmentdata[c][i][j])
