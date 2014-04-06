from DataManager import DataManager, NoSuchCategoryException
import string

class CLEFManager(DataManager):

    def __init__(self, *args, **kwargs):
        super(CLEFManager, self).__init__(*args, **kwargs)
        self.PATHS = {
            "BASE" : "/home/peter/thesis/data_ImageCLEF",
            "KEYPOINTS" : "/home/peter/thesis/data_ImageCLEF/features",
            "BOW" : "/home/peter/thesis/data_ImageCLEF/bow",
            "IMG" : "/home/peter/thesis/data_ImageCLEF/images",
            "CLASSIFIER" : "/home/peter/thesis/data_ImageCLEF/classifiers",
            "RESULTS" : "/home/peter/thesis/data_ImageCLEF/results",
            "LOGS" : "/home/peter/thesis/data_ImageCLEF/logs",
            "METADATA" : "/home/peter/thesis/data_ImageCLEF/metadata",
            "CATEGORY_LIST" : "/home/peter/thesis/data_ImageCLEF/metadata/concepts_2011.txt",
            "TRAIN_CATEGORIES" : "/home/peter/thesis/data_ImageCLEF/metadata/trainset_gt_annotations_corrected.txt",
            "TEST_CATEGORIES" : "/home/peter/thesis/data_ImageCLEF/metadata/testset_GT_annotations.txt",
        }

    # TODO: to super class?
    def _get_positive_samples(self, dataset, category):
        filenames = []
        if dataset in ["train", "all"]:
            filenames += self._extract_positive_samples(category, self.PATHS["TRAIN_CATEGORIES"])
        if dataset in ["test", "all"]:
            filenames += self._extract_positive_samples(category, self.PATHS["TEST_CATEGORIES"])
        return filenames

    def _extract_positive_samples(self, category, file_list):
        category_id = self._get_category_number(category)
        with open(file_list, "r") as f:
            return [line.split()[0] for line in f if line.split()[category_id + 1] == "1"]

    def _get_category_number(self, category_name):
        with open(self.PATHS["CATEGORY_LIST"], "r") as f:
            for line in f:
                nr, name = line.split()
                if string.lower(category_name) == string.lower(name):
                    return int(nr)

        raise NoSuchCategoryException("There is no category %s in the ImageCLEF dataset."
            % category_name)

    def get_image_names(self, dataset, category = None):
        img_names = []
        if dataset in ["train", "all"]:
            img_names += self._get_image_names(self.PATHS["TRAIN_CATEGORIES"])
        if dataset in ["test", "all"]:
            img_names += self._get_image_names(self.PATHS["TEST_CATEGORIES"])
        return img_names

    def _get_image_names(self, file_list):
        with open(file_list, "r") as f:
            return [line.split()[0].strip() for line in f]

    # Overwrite parent method to make category optional
    def build_sample_matrix(self, dataset, category = None):
        # TODO: Cache results?
        return super(CLEFManager, self).build_sample_matrix(dataset, category)
        
    def _build_sample_matrix(self, dataset, category = None, files = None):
        if not files:
            files = self.get_image_names(dataset)
        bow_files = self.get_bow_filenames(files)
        return self._stack_bow_vectors(bow_files)
        
    def _build_class_vector(self, dataset, category):
        bow_files = self.get_bow_filenames(self.get_image_names(dataset))
        positives = self.get_positive_samples(dataset, category)
        return self._generate_class_vector(category, bow_files, positives)