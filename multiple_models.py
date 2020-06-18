import src
import src.setup as config
from src.reading_data import Data
from src.classifier import Classifier
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# dt = Data()  # * Load the dataset in our expected format from the provided by AIDEA
cnn = Classifier(input_shape=(512, 512), batch_size=2)
conv_layers = ([(28, 3, 2), (32, 3, 1), (128, 3, 1)],
               [(66, 3, 2), (128, 3, 2)],
               [(96, 3, 2), (256, 5, 1), (384, 3, 1)])
dense_layers = ([1000, 100],
                [1000],
                [100])

for conv_ly in conv_layers:
    for dense_ly in dense_layers:
        cnn.set_custom_model(conv_layers=conv_ly, dense_layers=dense_ly)
        NAME = cnn.get_name()
        tensorboard = TensorBoard(log_dir=os.path.join(config.DATA_LOG, NAME))
        val_data = cnn.get_data_generator(
            config.DATA_VALIDATION_DIR, enhancement=False)
        train_data = cnn.get_data_generator(
            config.DATA_TRAIN_DIR, enhancement=False)
        cnn.train(gen_train=train_data, gen_val=val_data,
                  epochs=500, callbacks=tensorboard)
        print("done")
