import cv2
import numpy as np
from setup import *
import pandas as pd
import os
import shutil
from glob import glob


class Data:
    def __init__(self):
        self.dt_dict = dict()
        self.length = dict()
        for cat in ("train", "test"):
            csv_file = os.path.join(DATA_DIR, "{}.csv".format(cat))
            dt = pd.read_csv(csv_file).values
            self.dt_dict[cat] = dt
            self.length[cat] = dt.shape[0]
        self.sort_dataset()

    def get_sample(self, cat, idx=0):
        assert cat in ["train", "test"]
        assert idx < self.length[cat]
        image_file = os.path.join(DATA_TRAIN_DIR, classes[self.dt_dict[cat][idx, 1]], self.dt_dict[cat][idx, 0])
        return cv2.imread(image_file, 0), classes[self.dt_dict[cat][idx, 1]]

    def get_rand_sample(self, label):
        assert label in classes
        self.sort_dataset()
        image_file = np.random.choice(glob(os.path.join(DATA_TRAIN_DIR, label, "*.png")))

        return cv2.imread(image_file, 0), os.path.split(image_file)[1]

    @staticmethod
    def sort_dataset():
        assert os.path.isdir(DATA_TRAIN_DIR)
        # Reading csv file for training data
        dt_frame = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).values
        for idx, class_label in enumerate(classes):
            path = os.path.join(DATA_TRAIN_DIR, class_label)
            if not os.path.isdir(path):
                try:
                    os.mkdir(path)
                except OSError:
                    print("Creation of the directory %s failed" % path)
                else:
                    print("Successfully created the directory %s " % path)

            list_files_org = [os.path.join(DATA_TRAIN_DIR, file) for file in dt_frame[dt_frame[:, 1] == idx, 0]]
            list_files_destine = [os.path.join(DATA_TRAIN_DIR, class_label, file) for file in
                                  dt_frame[dt_frame[:, 1] == idx, 0]]
            [shutil.move(src, dst) for src, dst in zip(list_files_org, list_files_destine) if
             os.path.isfile(src)]
        print("Dataset has been sorted")

    @staticmethod
    def unsort_dataset():
        assert os.path.isdir(DATA_TRAIN_DIR)
        for class_dir in classes:
            list_files_org = glob(os.path.join(DATA_TRAIN_DIR, class_dir, "*.png"))
            list_files_destine = [os.path.join(DATA_TRAIN_DIR, os.path.split(file)[1]) for file in list_files_org]
            [shutil.move(src, dst) for src, dst in zip(list_files_org, list_files_destine) if
             os.path.isfile(src)]
        print("Dataset has been unsorted")

    def read_random_images_by_label(self, label, verbose=True):
        image, file_name = self.get_rand_sample(label)
        if verbose:
            image = cv2.putText(image, "Class: {}".format(label), (25, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                0.01,
                                cv2.LINE_4)
            image = cv2.putText(image, "File: {}".format(file_name), (25, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                0.01,
                                cv2.LINE_4)
        return image

    def read_random_image_by_cat(self, cat, verbose=True):
        idx = np.random.randint(0, self.length[cat], 1, np.int)[0]
        image, label = self.get_sample(cat, idx)
        if verbose:
            image = cv2.putText(image, "Class: {}".format(label), (25, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                0.01,
                                cv2.LINE_4)
            image = cv2.putText(image, "File: {}_{}.png".format(cat, idx), (25, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                0.01,
                                cv2.LINE_4)
        return image


if __name__ == '__main__':
    data = Data()
    id_class = 0
    while True:
        image1 = data.read_random_image_by_cat("train")
        image2 = data.read_random_images_by_label(label=classes[id_class])
        cv2.imshow("Random Image Selection", image1)
        cv2.imshow("Random Image Selection in class {}".format(classes[id_class]), image2)
        if cv2.waitKey(0) == 27:
            break
