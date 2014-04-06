from sklearn.externals import joblib
import time
from . import folder_name

class ClassifierLoader(object):
    """Handles saving and loading of trained classifiers transparently."""

    def __init__(self):
        super(ClassifierLoader, self).__init__()

    def dump_object(self, obj, classifier, category="", **kwargs):
        self.logger.info("Writing object to disk")
        t2 = time.time()
        try:
            folder = folder_name(self.datamanager.PATHS["CLASSIFIER"], category, classifier)
            if not os.path.isdir(folder):
                os.makedirs(folder)

            joblib.dump(obj, os.path.join(folder, fname), compress=3)
        except Exception as e:
            self.logger.error("Joblib failed: %s" % e)
        self.logger.info("%f seconds\n" % (time.time() - t2))

    def load_object(self, fname, category="", classifier=None):
        self.logger.info("Reading object from disk")
        t2 = time.time()
        if classifier == None:
            classifier = self.classifier

        try:
            folder = folder_name(self.datamanager.PATHS["CLASSIFIER"], category, classifier)
            if not os.path.isdir(folder):
                self.logger.info("Object's path doesn't exist")
                return None

            obj = joblib.load(os.path.join(folder, fname))
            self.logger.info("%f seconds\n" % (time.time() - t2))
            return obj
        except Exception as e:
            self.logger.error("Joblib failed: %s" % e)
            return None