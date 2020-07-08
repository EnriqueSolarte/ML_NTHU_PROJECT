import src.setup as st
from src.reading_data import Data
from src.classifier import Classifier
from tensorflow.keras.callbacks import TensorBoard
import os
from glob import glob
import tensorflow as tf
from PIL import Image
import numpy as np


def eval_model(cfg):
    cnn = Classifier(input_shape=cfg["shape"])
    cnn.set_default_AlexNet_Model()

    # if cfg["pre_trained"] is not None:
    #     dir_model = glob(os.path.join(st.DIR_LOG, cfg["log"], "*" + cfg["pre_trained"] + "*"))
    #     assert len(dir_model) == 1, "the len of chekpoint list is {}".format(len(dir_model))
    #     cnn.model.load_weights(os.path.join(dir_model[0], "cp-model.ckpt"))

    assert cfg["pre_trained"] is not None
    dir_model = glob(os.path.join(st.DIR_LOG, cfg["log"], "*" + cfg["pre_trained"] + "*"))
    assert len(dir_model) == 1, "the len of chekpoint list is {}".format(len(dir_model))
    model_file = os.path.join(dir_model[0], "cp-model.ckpt")
    # model_file = os.path.join(dir_model[0], "'model-086-1.000000.ckpt")
    cnn.model.load_weights(model_file)

    dt = Data()
    sum_pred = np.zeros(6,)
    lenght_data = np.zeros(6,)
    for idx in range(dt.length['train']):
        image_file, label_gt = dt.get_image_file('train', idx, encode_label=True)
        label_est = cnn.model_predict(image_file, cfg["dip"])
        lenght_data[label_gt] += 1
        if label_est == label_gt:
            sum_pred[label_gt] += 1

        acc = sum_pred / lenght_data
        print("{} - Acc:  {}".format((label_gt, label_est), acc))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    pre_trained = ["17.04", "16.41", "00.56.01"]
    config = dict(shape=(224, 224),
                  dip=True,
                  log="RUNNING",
                  #   log="SUCCESSFULLY_TRAINING",
                  pre_trained="08-12.57")

    eval_model(cfg=config)
