import cv2
import numpy as np
from setup import *
import pandas as pd


def read_random_images_by_label(label, verbose=True):
    data = Data()
    image, idx = data.get_rand_image(label)
    if verbose:
        image = cv2.putText(image, "Label: {} - train_{}.png".format(label, idx), (25, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            0.01,
                            cv2.LINE_4)
    return image


def read_random_image_by_cat(_cat, verbose=True):
    _data = Data()
    idx = np.random.randint(0, _data.length[_cat], 1, np.int)[0]
    image, label = _data.get_image_idx(_cat, idx)
    if verbose:
        image = cv2.putText(image, "Label: {} - {}_{}.png".format(label, _cat, idx), (25, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            0.01,
                            cv2.LINE_4)
    return image


class Data:
    def __init__(self):
        self.data_file_list = dict()
        self.length = dict()
        for cat in ("train", "test"):
            csv_file = os.path.join(DATA_DIR, "{}.csv".format(cat))
            dt = pd.read_csv(csv_file)
            self.data_file_list[cat] = ([os.path.join(DATA_DIR, "{}_images".format(cat), image) for image in
                                         tuple(dt["ID"].values)],
                                        list(dt["Label"].values))
            self.length[cat] = dt["Label"].size

    def get_image_idx(self, cat, idx=0):
        """
        Return the image-idx in the category defined by cat
        cat: [train] [test]
        """
        assert idx < len(self.data_file_list[cat][0])
        return cv2.imread(self.data_file_list[cat][0][idx]), np.asarray(self.data_file_list[cat][1][idx])

    def get_rand_image(self, _label):
        assert _label in range(6)
        mask = np.where(np.asarray(self.data_file_list["train"][1]) == _label)
        _idx = np.random.randint(0, len(mask[0]), 1)[0]
        return cv2.imread(self.data_file_list['train'][0][mask[0][_idx]]), mask[0][_idx]


if __name__ == '__main__':
    while True:
        image1 = read_random_image_by_cat("train")
        image2 = read_random_images_by_label(label=0)
        cv2.imshow("Image1", image1)
        cv2.imshow("Image2", image2)
        if cv2.waitKey(0) == 27:
            break
