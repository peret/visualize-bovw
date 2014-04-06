"""Helper script to split Caltech101 in test
and training set.

Set TEST_SET_RATE accordingly and run
in Caltech101 folder.
"""

import os
import random
from math import ceil



if __name__ == '__main__':
    TEST_SET_RATE = 0.5
    for category_dir in os.listdir("."):
        if os.path.isdir(category_dir):
            images = os.listdir(category_dir)
            test_set_size = int(ceil(len(images) * TEST_SET_RATE))
            test_images = random.sample(images, test_set_size)
            os.mkdir(os.path.join(category_dir, "test"))
            for test_img in test_images:
                os.rename(  os.path.join(category_dir, test_img),
                            os.path.join(category_dir, "test", test_img))