import numpy as np
import os
import util
from util.data import storage
from sklearn.decomposition import PCA

class NoSuchCategoryException(Exception):
    pass

class InvalidDatasetException(Exception):
    pass

class DataManager(object):

    def __init__(self, exclude_feature=None):
        # Dictionary containing paths to the relevant
        # data directories/files
        self.PATHS = {}
        self.pca = None
        self.exclude_feature = exclude_feature

    def change_base_path(self, base):
        old_base = self.PATHS["BASE"]
        for k,v in self.PATHS.items():
            self.PATHS[k] = v.replace(old_base, base)

    def get_positive_samples(self, dataset, category):
        """
        Get filenames of the positive sample images for the given
        category. The dataset can be either of "train", "test" or "all".
        """
        if not dataset in ["train", "test", "all"]:
            raise InvalidDatasetException("The dataset specifier should be either of 'train', 'test' or 'all'.")
        return self._get_positive_samples(dataset, category)

    def _get_positive_samples(self, dataset, category):
        raise NotImplementedError

    def build_sample_matrix(self, dataset, category, files=None):
        if not dataset in ["train", "test", "all"]:
            raise InvalidDatasetException("The dataset specifier should be either of 'train', 'test' or 'all'.")
        samples = self._build_sample_matrix(dataset, category, files)
        if self.pca:
            samples = self.pca.transform(samples)
        return samples

    def _build_sample_matrix(self, dataset, category):
        raise NotImplementedError

    def _stack_bow_vectors(self, bow_files):
        # log("Loading matrices")
        bow_vectors = [storage.load_matrix(os.path.join(self.PATHS["BOW"], f)) for f in bow_files]
        # log("Building sample matrix")

        # build a n_samples x n_features matrix for sklearn by concatenating the BoW-vectors
        matrix = np.vstack(bow_vectors)
        # log("Vocabulary size: %d" % matrix.shape[1])
        return matrix

    def build_class_vector(self, dataset, category):
        if not dataset in ["train", "test", "all"]:
            raise InvalidDatasetException("The dataset specifier should be either of 'train', 'test' or 'all'.")
        return self._build_class_vector(dataset, category)

    # TODO: Overloaded method somewhat unclear.
    def _build_class_vector(self, dataset, category):
        raise NotImplementedError

    def _generate_class_vector(self, category, bow_files, positives):
        positives = [util.bow_name(p) for p in positives]
        # log("Number of positive examples: ", len(positives))

        y = np.zeros(len(bow_files), dtype=np.int32)
        indices = np.array([f in positives for f in bow_files], dtype=np.bool)
        y[indices] = 1
        return y

    def get_bow_filenames(self, file_names):
        files = set(os.listdir(self.PATHS["BOW"]))
        bows = [util.bow_name(f) for f in file_names]
        return [b for b in bows if b in files]

    def use_pca(self, *args, **kwargs):
        samples = self.build_sample_matrix("train", None)
        self.pca = PCA(*args, **kwargs)
        self.pca.fit(samples)