import cv2
import numpy as np
import setup as st
import pandas as pd
import os
import shutil
from glob import glob
import cv2


class Data:
    def __init__(self, validation_ratio=None):
        self.dt_dict = dict()
        self.length = dict()
        self.isSorted = False
        for cat in ("train", "test"):
            csv_file = os.path.join(st.DATA_DIR, "{}.csv".format(cat))
            dt = pd.read_csv(csv_file).values
            self.dt_dict[cat] = dt
            self.length[cat] = dt.shape[0]

        if validation_ratio is not None:
            self.unsort_dataset()
            self.sort_dataset(split_data_ratio=validation_ratio)

    def sort_dataset(self, split_data_ratio=0.5):
        assert os.path.isdir(st.DATA_TRAIN_DIR)
        # Reading csv file for training data
        dt_frame = pd.read_csv(os.path.join(st.DATA_DIR, "train.csv")).values
        for idx, class_label in enumerate(st.CLASSES):
            for dir_ in [st.DATA_TRAIN_DIR, st.DATA_VALIDATION_DIR]:
                path = os.path.join(dir_, class_label)
                if not os.path.isdir(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                    except OSError:
                        print("Creation of the directory %s failed" % path)
                    else:
                        print("Successfully created the directory %s " % path)

            # * All files per category
            files_per_cat = [file for file in dt_frame[dt_frame[:, 1] == idx, 0]]
            np.random.shuffle(files_per_cat)
            idx_split = int(len(files_per_cat) * split_data_ratio)
            for list_images, dir_ in zip([files_per_cat[idx_split:], files_per_cat[0:idx_split]],
                                         [st.DATA_TRAIN_DIR, st.DATA_VALIDATION_DIR]):
                
                if split_data_ratio == 0:
                    if dir_ == st.DATA_VALIDATION_DIR:
                        continue
                    
                list_files_destine = [os.path.join(dir_, class_label, file) for file in list_images]
                list_files_origin = [os.path.join(st.DATA_TRAIN_DIR, file) for file in list_images]
                [shutil.move(src, dst)
                 for src, dst in zip(list_files_origin, list_files_destine) if
                 os.path.isfile(src)]
        print("Dataset has been sorted")
        self.isSorted = True

    def unsort_dataset(self):
        assert os.path.isdir(st.DATA_TRAIN_DIR)
        for class_dir in st.CLASSES:
            list_files_org = glob(os.path.join(st.DATA_TRAIN_DIR, class_dir, "*.png"))
            list_files_org.extend(glob(os.path.join(st.DATA_VALIDATION_DIR, class_dir, "*.png")))
            list_files_destine = [os.path.join(st.DATA_TRAIN_DIR, os.path.split(file)[1]) for file in list_files_org]
            [shutil.move(src, dst) for src, dst in zip(list_files_org, list_files_destine) if
             os.path.isfile(src)]
        print("Dataset has been unsorted")
        self.isSorted = False

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

    @staticmethod
    def get_rand_sample(label):
        assert label in st.CLASSES
        image_file = np.random.choice(glob(os.path.join(st.DATA_TRAIN_DIR, label, "*.png")))
        return cv2.imread(image_file, 0), os.path.split(image_file)[1]

    def get_sample(self, cat, idx=0):
        image_file, label = self.get_image_file(cat, idx)
        return cv2.imread(image_file, 0), label

    def get_image_file(self, cat, idx=0, encode_label=False):
        assert cat in ["train", "test"]
        assert idx < self.length[cat]
        image_file = os.path.join(st.DATA_TRAIN_DIR, st.CLASSES[self.dt_dict[cat][idx, 1]], self.dt_dict[cat][idx, 0])
        if not os.path.isfile(image_file):
            image_file = os.path.join(st.DATA_VALIDATION_DIR, st.CLASSES[self.dt_dict[cat][idx, 1]],
                                      self.dt_dict[cat][idx, 0])
        if encode_label:
            label = self.dt_dict[cat][idx, 1]
        else:
            label = st.CLASSES[self.dt_dict[cat][idx, 1]]
        return image_file, label


if __name__ == '__main__':
    data = Data()
    id_class = 0
    while True:
        image1 = data.read_random_image_by_cat("train")
        image2 = data.read_random_images_by_label(label=st.CLASSES[id_class])
        cv2.imshow("Random Image Selection", image1)
        cv2.imshow("Random Image Selection in class {}".format(st.CLASSES[id_class]), image2)
        if cv2.waitKey(0) == 27:
            break
