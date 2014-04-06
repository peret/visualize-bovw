import datamanagers
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

class MyTestDataManager(datamanagers.CLEFManager):
    def __init__(self, *args, **kwargs):
        super(MyTestDataManager, self).__init__(*args, **kwargs)
        self.change_base_path(os.path.join(BASE_PATH, "testdata"))