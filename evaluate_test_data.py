import src.setup as st
from src.reading_data import Data
from src.classifier import Classifier
from tensorflow.keras.callbacks import TensorBoard
import os
from glob import glob
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import csv


def red_csv_file():
    csv_file = os.path.join(st.DATA_DIR, "test.csv")
    dt = pd.read_csv(csv_file).values[:, 0]
    return dt


def eval_model(cfg):
    cnn = Classifier(input_shape=cfg["shape"])
    cnn.set_default_AlexNet_Model()

    assert cfg["pre_trained"] is not None
    dir_model = glob(os.path.join(st.DIR_LOG, cfg["log"], "*" + cfg["pre_trained"] + "*"))
    assert len(dir_model) == 1, "the len of chekpoint list is {}".format(len(dir_model))
    model_file = os.path.join(dir_model[0], "cp-model.ckpt")
    # model_file = os.path.join(dir_model[0], "'model-086-1.000000.ckpt")
    cnn.model.load_weights(model_file)

    dt = red_csv_file()
    filename = os.path.join(st.DATA_DIR, "eval_test_data_{}_{}.csv".format(cfg["pre_trained"], cfg["extra"]))
    print(filename)
    with open(filename, '+w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(("ID", "Label"))
    for idx in range(len(dt)):
        image_file = os.path.join(st.DATA_TEST_DIR, dt[idx])
        assert os.path.isfile(image_file)
        label_est = cnn.model_predict(image_file, cfg["dip"])

        with open(filename, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow((dt[idx], label_est))

        print("{0:} - progress:  {1:.1f}".format(idx, 100*idx/len(dt)))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    config = dict(shape=(224, 224),
                  dip=True,
                  log="RUNNING",
                  #   log="finals",
                  #   log="SUCCESSFULLY_TRAINING",
                  pre_trained="08-12.57",
                  extra="final")

    eval_model(cfg=config)
