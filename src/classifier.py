from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import setup as config
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from reading_data import Data
from filters import he, random_he
from datetime import datetime
import os 

class Classifier:
    def __init__(self, input_shape, batch_size=2):
        self.name = "{}x{}-{}".format(input_shape[0], input_shape[1], batch_size)
        self.model = None
        self.input_shape = input_shape
        self.image_width = input_shape[0]
        self.image_height = input_shape[1]
        self.batch_size = batch_size

    def get_name(self):
        data_time_obj = datetime.now()
        return self.name + data_time_obj.strftime("%y.%m.%d-%H.%M.%s")

    def set_default_AlexNet_Model(self):
        self.name += "-AlexNet-{}x{}-".format(self.image_height, self.image_width)
        _input = Input(shape=(self.image_height, self.image_width, 1))

        # ! 1st convolution layer
        x = Conv2D(filters=96,
                   kernel_size=11,
                   strides=4,
                   padding='same',
                   activation='relu')(_input)

        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        # ! 2nd convolution layer
        x = Conv2D(filters=256,
                   kernel_size=5,
                   padding='same',
                   activation='relu')(x)

        x = BatchNormalization()(x)
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
                   activation='relu')(x)

        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Flatten()(x)
        x = Dense(units=4096, activation='relu')(x)
        x = Dense(units=4096, activation='relu')(x)
        x = Dropout(rate=0.5)(x)

        output = Dense(units=6, activation='softmax')(x)

        self.model = Model(inputs=_input, outputs=output)

    def set_custom_model(self, conv_layers, dense_layers):

        _input = Input(shape=(self.image_height, self.image_width, 1))

        self.name += "[C{}.{}.{}]".format(conv_layers[0][0], conv_layers[0][1], conv_layers[0][2])
        x = Conv2D(filters=conv_layers[0][0],
                   kernel_size=conv_layers[0][1],
                   strides=conv_layers[0][2],
                   padding='same',
                   activation='relu')(_input)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        for layer_ in conv_layers[1:]:
            self.name += "[C{}.{}.{}]".format(layer_[0], layer_[1], layer_[2])
            x = Conv2D(filters=layer_[0],
                       kernel_size=layer_[1],
                       strides=layer_[2],
                       padding='same',
                       activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Flatten()(x)

        for layer_ in dense_layers:
            self.name += "[D{}]".format(layer_)
            if layer_ is not None or layer_ != 0:
                x = Dense(units=layer_, activation='relu')(x)

        x = Dropout(rate=0.5)(x)

        output = Dense(units=6, activation='softmax')(x)

        self.model = Model(inputs=_input, outputs=output)

    def get_data_generator(self, path_dir, enhancement=False, random_boost=False):
        assert os.path.isdir(path_dir)
        if not random_boost:
            self.name += "he_{}.".format(enhancement)
            if enhancement:
                dt_generator = ImageDataGenerator(rescale=1 / 255,
                                                  horizontal_flip=True,
                                                  vertical_flip=True,
                                                  preprocessing_function=he)
            else:
                dt_generator = ImageDataGenerator(rescale=1 / 255,
                                                  horizontal_flip=True,
                                                  vertical_flip=True)
        else:
            self.name += "rand_he_{}.".format(random_boost)
            dt_generator = ImageDataGenerator(rescale=1 / 255,
                                              horizontal_flip=True,
                                              vertical_flip=True,
                                              preprocessing_function=random_he)

        return dt_generator.flow_from_directory(path_dir,
                                                target_size=(self.image_width, self.image_height),  # input image size
                                                batch_size=self.batch_size,  # batch size
                                                classes=config.classes)

        # dt_generator = ImageDataGenerator(rescale=1 / 255)
        # return dt_generator.flow_from_directory(path_dir,
        #                                         target_size=(self.image_width, self.image_height),  # input image size
        #                                         batch_size=self.batch_size,  # batch size
        #                                         classes=classes)

    def train(self, gen_train, gen_val, epochs, callbacks):
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(
            gen_train,
            steps_per_epoch=np.floor(gen_train.n / self.batch_size),
            epochs=epochs,
            validation_data=gen_val,
            validation_steps=np.floor(gen_val.n / self.batch_size),
            callbacks=[callbacks])

    def model_predict(self, image_path):
        assert os.path.isfile(image_path)
        # Loads image into PIL
        img = load_img(image_path, grayscale=False, color_mode="rgb", target_size=(224, 224), interpolation="nearest")
        # Converts PIL to np array
        img = img_to_array(img)
        # creates a batch for single image
        img = np.array([img]) / 255
        # Finds the index of highest value after prediction [0 0 0 1 0 0]
        predict = self.model.predict(img)
        result = np.argmax(predict)
        return result


if __name__ == '__main__':
    dt = Data()  # ! Load the dataset in our expected format from the provided by AIDEA
    cnn = Classifier((224, 224))
    val_data = cnn.get_data_generator(config.DATA_VALIDATION_DIR, enhancement=True)
    train_data = cnn.get_data_generator(config.DATA_TRAIN_DIR, enhancement=True)
    cnn.train(gen_train=train_data, gen_val=val_data, epochs=10, callbacks=None)
    print("done")
