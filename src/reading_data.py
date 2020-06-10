import cv2
import numpy as np
from setup import *
import pandas as pd


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
        return cv2.imread(self.data_file_list[cat][0][idx]), self.data_file_list[cat][1][idx]


if __name__ == '__main__':
    data = Data()
    while True:
        idx = np.random.randint(0, data.length["train"], 1, np.int)[0]
        image, label = data.get_image_idx("train", idx)
        cv2.imshow("Image cat:{}".format(label), image)
        cv2.waitKey(0)
