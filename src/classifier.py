from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras import Model, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy as cce
from tensorflow.keras.backend import get_value
import setup as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from reading_data import Data
from filters import he, random_he
from datetime import datetime
import os


class Classifier:
    def __init__(self, input_shape, batch_size=2):
        data_time_obj = datetime.now()
        self.model = None
        self.input_shape = input_shape
        self.image_width = input_shape[0]
        self.image_height = input_shape[1]
        self.batch_size = batch_size
        self.random_boost = False
        self.log_name = data_time_obj.strftime("%y.%m.%d-%H.%M.%S")
        self.callbacks = []
        self.lr = dict(lr=1e-3,
                       decay_steps=200,
                       decay_rate=0.5)

        self.mask_in = dict(model0=np.array((1, 1, 1, 1, 1, 1)),
                            model1=np.array((0.2, 0.5, 1, 0.5, 0.5, 0.2)),
                            model2=np.array((0.1, 0.2, 0.3, 1, 0.3, 0.1)),
                            model3=np.array((0.99, 0.95, 0.93, 0.85, 0.93, 0.99)),
                            model4=np.array((0.99, 0.95, 0.93, 1.0, 0.93, 0.99)),
                            model5=np.array((0.5, 0.5, 0.5, 0.99, 0.5, 0.5)),
                            model6=np.array((0.0, 0.0, 0.0, 0.99, 0.0, 0.0)),
                            model7=np.array((0.0, 0.5, 1.0, 1.0, 0.5, 0.0)))
        self.model_select = "model0"

    def set_log_name(self, cfg):

        key_list = cfg.keys()
        for key in key_list:
            if key == "pre_trained":
                continue
            if key == "arch":
                continue
            self.log_name += "-" + key + str(cfg[key])

    def set_default_AlexNet_Model(self):
        _input = Input(shape=(self.image_height, self.image_width, 1))

        # ! 1st convolution layer
        x = Conv2D(filters=96,
                   kernel_size=11,
                   strides=4,
                   padding='same',
                   use_bias=False)(_input)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        # ! 2nd convolution layer
        x = Conv2D(filters=256,
                   kernel_size=5,
                   padding='same',
                   use_bias=False)(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        # ! 3rd convolution layer
        x = Conv2D(filters=384,
                   kernel_size=3,
                   padding='same',
                   activation='relu')(x)

        # ! 4th convolution layer
        x = Conv2D(filters=384,
                   kernel_size=3,
                   padding='same',
                   activation='relu')(x)

        # ! 5th convolution layer
        x = Conv2D(filters=256,
                   kernel_size=3,
                   padding='same',
                   use_bias=False)(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Flatten()(x)
        x = Dense(units=4096, activation='relu')(x)
        x = Dense(units=4096, activation='relu')(x)
        x = Dropout(rate=0.5)(x)

        output = Dense(units=6, activation='softmax')(x)

        self.model = Model(inputs=_input, outputs=output)

    def set_custom_model(self, conv_layers, dense_layers):

        _input = Input(shape=(self.image_height, self.image_width, 1))

        x = Conv2D(filters=conv_layers[0][0],
                   kernel_size=conv_layers[0][1],
                   strides=conv_layers[0][2],
                   padding='same',
                   activation='relu')(_input)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        for layer_ in conv_layers[1:]:
            x = Conv2D(filters=layer_[0],
                       kernel_size=layer_[1],
                       strides=layer_[2],
                       padding='same',
                       activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Flatten()(x)

        for layer_ in dense_layers:
            if layer_ is not None or layer_ != 0:
                x = Dense(units=layer_, activation='relu')(x)

        x = Dropout(rate=0.5)(x)

        output = Dense(units=6, activation='softmax')(x)

        self.model = Model(inputs=_input, outputs=output)

    def get_data_generator(self, path_dir, dip_filter=False):
        assert os.path.isdir(path_dir), "Path does not exist: {}".format(path_dir)
        if dip_filter:
            dt_generator = ImageDataGenerator(rescale=1 / 255,
                                              horizontal_flip=True,
                                              vertical_flip=True,
                                              preprocessing_function=self.dip_filtering)
        else:
            dt_generator = ImageDataGenerator(rescale=1 / 255,
                                              horizontal_flip=True,
                                              vertical_flip=True)

        return dt_generator.flow_from_directory(path_dir,
                                                target_size=(self.image_width, self.image_height),  # input image size
                                                batch_size=self.batch_size,  # batch size
                                                classes=st.CLASSES,
                                                shuffle=True)

    def train(self, gen_train, gen_val, epochs):
        checkpoint_path = os.path.join(st.DIR_LOG, "RUNNING", self.log_name,
                                       #    "'model-{epoch:03d}-{accuracy:03f}.ckpt",
                                       "cp-model.ckpt")
        # ! Create callback checkpoint
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                       save_weights_only=True,
                                                                       monitor='val_loss',
                                                                       mode='min',
                                                                       save_best_only=True,
                                                                       save_freq='epoch',
                                                                       verbose=1)

        self.callbacks.append(model_checkpoint_callback)

        lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr["lr"],
                                                            decay_steps=self.lr["decay_steps"],
                                                            decay_rate=self.lr["decay_rate"])

        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule,
                                                             momentum=self.lr["momentum"],
                                                             nesterov=False),
                           loss=self.weighted_loss(self.mask_in[self.model_select]), metrics=['accuracy'])
        # self.model.compile(optimizer=tf.keras.optimizers.Adam(),
        #                    loss=self.custom_loss(self.mask_in[self.model_select]), metrics=['accuracy'])
        self.model.fit(gen_train,
                       steps_per_epoch=np.floor(gen_train.n / self.batch_size),
                       epochs=epochs,
                       validation_data=gen_train,
                       validation_steps=np.floor(gen_train.n / self.batch_size),
                       callbacks=self.callbacks)

    def model_predict(self, image_path, dip_filter=False):
        assert os.path.isfile(image_path)
        # Loads image into PIL
        img = load_img(image_path, grayscale=True, color_mode="rgb", target_size=(224, 224), interpolation="nearest")
        # Converts PIL to np array
        if dip_filter:
            img = he(img_to_array(img))
        else:
            img = img_to_array(img)
        # creates a batch for single image
        img = np.array([img]) / 255
        # Finds the index of highest value after prediction [0 0 0 1 0 0]
        predict = self.model.predict(img)
        # if np.max(predict) < 0.5:
        #     return -1
        return np.argmax(predict)

    def dip_filtering(self, image):
        if self.random_boost:
            return random_he(image)
        else:
            return he(image)

    def weighted_loss(self, mask):
        def loss(y_true, y_pred):
            return cce(y_pred=y_pred*mask, y_true=y_true)
        return loss


if __name__ == '__main__':
    dt = Data()  # ! Load the dataset in our expected format from the provided by AIDEA
    cnn = Classifier((224, 224))
    cnn.set_default_AlexNet_Model()
    val_data = cnn.get_data_generator(st.DATA_VALIDATION_DIR, dip_filter=True)
    train_data = cnn.get_data_generator(st.DATA_TRAIN_DIR, dip_filter=True)
    cnn.train(gen_train=train_data, gen_val=val_data, epochs=1)
    print("done")
