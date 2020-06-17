import src
from src.setup import *
from reading_data import Data
from classifier import Classifier
from tensorflow.keras.callbacks import TensorBoard

dt = Data()  # * Load the dataset in our expected format from the provided by AIDEA
cnn = Classifier(input_shape=(512, 512))
cnn.set_custom_model(conv_layers=[[32, 3, 2], [64, 3, 2]], dense_layers=[1000])
NAME = cnn.get_name()
tensorboard = TensorBoard(log_dir=os.path.join(DATA_LOG, NAME))

val_data = cnn.get_data_generator(DATA_VALIDATION_DIR, enhancement=True)
train_data = cnn.get_data_generator(DATA_TRAIN_DIR, enhancement=True)
cnn.train(gen_train=train_data, gen_val=val_data, epochs=10, callbacks=tensorboard)
print("done")
