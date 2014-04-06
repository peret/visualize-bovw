#import data
import os
import string
import numpy as np
from itertools import izip

def map_to_old_params(p):
    assert p['base_estimator__compute_importances'] is None
    p['base_estimator__compute_importances'] = False
    assert p['base_estimator__min_density'] is None
    p['base_estimator__min_density'] = 0.1
    del p['base_estimator__splitter']
    del p['random_state']
    return p

def params_to_path(p):
    t = "%s__%s"
    # print only the name of the base estimator class
    if "base_estimator__splitter" in p:
        p = map_to_old_params(p)
    return "/".join([t % (k, v) if k!="base_estimator"
                     else t % (k, class_name(v)) for k, v in sorted(p.items())])

def folder_name(base, category, clf):
    return os.path.join(base, class_name(clf), category, params_to_path(clf.get_params()))

def class_name(x):
    return x.__class__.__name__

def bow_name(path):
    return os.path.splitext(path)[0] + ".descr_bowdescr.bin"

def keypoints_name(path):
    return os.path.splitext(path)[0] + ".kps"

def descriptors_name(path):
    return os.path.splitext(path)[0] + ".descr.mat"

def cluster_name(path):
    return os.path.splitext(path)[0] + ".descr_indices.bin"

def generate_evaluation_summary(grids, fname):
    with open(fname, "w") as table:
        table.write(",".join(["category", "estimators", "max_depth", "accuracy", "average_precision", "training images"]))
        table.write("\n")

        for grid in grids:
            classifier = grid.grid_search_obj.best_estimator_
            grid.classifier = classifier
            acc, avp = grid.evaluate_test_set(grid.category)
            img_count = len(grid.datamanager._get_positive_samples("train", grid.category))
            table.write(",".join([grid.category, str(classifier.n_estimators), str(classifier.base_estimator.max_depth), str(acc), str(avp), str(img_count)]))
            table.write("\n")