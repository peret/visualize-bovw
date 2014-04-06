from DataManager import DataManager, NoSuchCategoryException
from sklearn.decomposition import PCA
import os
import numpy as np

class CaltechManager(DataManager):

    def __init__(self, *args, **kwargs):
        super(CaltechManager, self).__init__(*args, **kwargs)
        self.PATHS = {
            "BASE" : "/home/peter/thesis/data_caltech101",
            "KEYPOINTS" : "/home/peter/thesis/data_caltech101/features",
            "BOW" : "/home/peter/thesis/data_caltech101/bow",
            "IMG" : "/home/peter/thesis/data_caltech101/images",
            "CLASSIFIER" : "/home/peter/thesis/data_caltech101/classifiers",
            "RESULTS" : "/home/peter/thesis/data_caltech101/results",
            "LOGS" : "/home/peter/thesis/data_caltech101/logs",
            "CATEGORIES_DIR" : "/home/peter/thesis/data_caltech101/101_ObjectCategories"
        }
        self.BACKGROUND = "BACKGROUND_Google"

    def _get_positive_samples(self, dataset, category):
        path = os.path.join(self.PATHS["CATEGORIES_DIR"], category)
        fnames = []
        if not os.path.isdir(path):
            raise NoSuchCategoryException("There is no category %s in the Caltech101 dataset."
                % category)

        if dataset in ["train", "all"]:
            fnames += self._get_all_files(category, path)
            
        if dataset in ["test", "all"]:
            path = os.path.join(path, "test")
            fnames += self._get_all_files(category, path)
        return fnames

    def get_image_names(self, dataset, category):
        img_names = []
        for c in [category, self.BACKGROUND]:
        # for c in [category]:
            if dataset in ["train", "all"]:
                path = os.path.join(self.PATHS["CATEGORIES_DIR"], c)
                img_names += self._get_all_files(c, path)

            if dataset in ["test", "all"]:
                path = os.path.join(self.PATHS["CATEGORIES_DIR"], c, "test")
                img_names += self._get_all_files(c, path)
        return img_names

    def _get_all_files(self, category, path):
        if os.path.isdir(path):
            return ["%s_%s" % (category, f) for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]
        else:
            raise NoSuchCategoryException("There is no category %s in the Caltech101 dataset." % category)

    def _build_sample_matrix(self, dataset, category, files=None):
        if not files:
            files = self.get_image_names(dataset, category)
        bow_files = self.get_bow_filenames(files)
        if not self.exclude_feature:
            return self._stack_bow_vectors(bow_files)
        else:
            return np.delete(self._stack_bow_vectors(bow_files), self.exclude_feature, 1)

    def _build_class_vector(self, dataset, category):
        bow_files = self.get_bow_filenames(self.get_image_names(dataset, category))
        positives = self.get_positive_samples(dataset, category)
        return self._generate_class_vector(category, bow_files, positives)

    def build_complete_sample_matrix(self, dataset):
        if dataset != "train":
            raise NotImplementedError

        img_names = []
        for c in os.listdir(self.PATHS["CATEGORIES_DIR"]):
            path = os.path.join(self.PATHS["CATEGORIES_DIR"], c)
            if os.path.isdir(path):
                # print c, path
                img_names += self._get_all_files(c, path)

        if not self.exclude_feature:
            return self._stack_bow_vectors(self.get_bow_filenames(img_names))
        else:
            return np.delete(self._stack_bow_vectors(self.get_bow_filenames(img_names)), self.exclude_feature, 1)

    # TODO: to parent class
    def use_pca(self, *args, **kwargs):
        samples = self.build_complete_sample_matrix("train")
        self.pca = PCA(*args, **kwargs)
        self.pca.fit(samples)