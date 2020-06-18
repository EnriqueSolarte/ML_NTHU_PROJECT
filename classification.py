import src.setup as config
from src.reading_data import Data
from src.classifier import Classifier
from tensorflow.keras.callbacks import TensorBoard
import os

# dt = Data()  # * Load the dataset in our expected format from the provided by AIDEA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cnn = Classifier(input_shape=(512, 512))
cnn.set_default_AlexNet_Model()

NAME = cnn.get_name()
tensorboard = TensorBoard(log_dir=os.path.join(config.DATA_LOG, NAME))

val_data = cnn.get_data_generator(config.DATA_VALIDATION_DIR, enhancement=True)
train_data = cnn.get_data_generator(config.DATA_TRAIN_DIR, enhancement=True)
cnn.train(gen_train=train_data, gen_val=val_data, epochs=500, callbacks=tensorboard)
print("done")
