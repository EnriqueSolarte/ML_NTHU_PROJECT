import os
import numpy as np
import csv
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


def AlexNetModel():
    input = Input(shape=(224, 224, 3))

    x = Conv2D(filters=96,
               kernel_size=11,
               strides=4,
               padding='same',
               activation='relu')(input)  # 1st convolutional layer

    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(filters=256,
               kernel_size=5,
               padding='same',
               activation='relu')(x)  # 2nd convolutional layer

    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(filters=384,
               kernel_size=3,
               padding='same',
               activation='relu')(x)  # 3rd convolutional layer

    x = Conv2D(filters=384,
               kernel_size=3,
               padding='same',
               activation='relu')(x)  # 4th convolutional layer

    x = Conv2D(filters=256,
               kernel_size=3,
               padding='same',
               activation='relu')(x)  # 5th convolutional layer

    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Flatten()(x)
    x = Dense(units=4096, activation='relu')(x)
    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)

    output = Dense(units=6, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def Images_into_batches(train_dir, test_dir, val_dir, batch_size):
    image_width, image_height = 224, 224

    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255)
    val_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        # color_mode='grayscale',#inpput iameg: gray
        target_size=(image_width, image_height),  # input image size
        batch_size=batch_size,  # batch size
        # color_mode='rgb',
        # class_mode='categorical',#categorical: one-hot encoding format class label
        classes=['normal', 'void', 'horizontal_defect', 'vertical_defect', 'edge_defect', 'particle'])

    testing_generator = test_datagen.flow_from_directory(
        test_dir,
        # color_mode='grayscale',
        target_size=(image_width, image_height),
        batch_size=1,
        # batch_size=sum(len(files) for _, _, files in os.walk(test_dir))-1,
        # color_mode='rgb',
        # class_mode='categorical',
        classes=['normal', 'void', 'horizontal_defect', 'vertical_defect', 'edge_defect', 'particle'])

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        # color_mode='grayscale',
        target_size=(image_width, image_height),
        batch_size=batch_size,
        # color_mode='rgb',
        # class_mode='categorical',
        classes=['normal', 'void', 'horizontal_defect', 'vertical_defect', 'edge_defect', 'particle'])

    return train_generator, testing_generator, val_generator


def model_train(model, train_generator, testing_generator, epochs, batch_size):
    model.fit_generator(
        train_generator,
        steps_per_epoch=np.floor(train_generator.n / batch_size),
        epochs=epochs,
        validation_data=testing_generator,
        validation_steps=np.floor(testing_generator.n / batch_size))

    return model


def model_predict(model, test_batch):
    results = []
    f_path = test_batch.filenames
    # step = sum(len(files) for _, _, files in os.walk(test_batch.directory))-1
    predict = model.predict_generator(test_batch, steps=len(f_path), verbose=0)
    for i in range(0, len(predict)):
        max_predict = np.argmax(predict[i])
        # Stores the file names by excluding the directory names and stores the predicted results 
        results.append([(f_path[i].split("/"))[-1], max_predict])

    '''
    results =[]
    predict = []
    for r, d, f in os.walk(test_dir): # Reads all file in folder
        for file in f:
            if '.png' in file:   
                # Loads image into PIL
                img = load_img(os.path.join(r, file),grayscale=False, color_mode="rgb", target_size=(224,224),interpolation="nearest")
                # Converts PIL to np array
                img = img_to_array(img)
                # creates a batch for single image
                img = np.array([img])
                # Finds the index of highest value after prediction [0 0 0 1 0 0]
                predict.append(model.predict(img))
                result = (np.argmax(predict[-1]))
                results.append([file,result])
                '''
    return results


def calculate_accuracy(predicted, test_csv_path):
    test = []
    with open(test_csv_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            test.append(row)
    correct_predict = 0
    for j in range(0, len(predicted)):
        test_list = [i[0] for i in test]
        to_find = (predicted[j][0])
        pos = test_list.index(to_find)
        ground_truth = test[pos][1]
        if ground_truth == predicted[j][1]:
            correct_predict = correct_predict + 1
    accuracy = correct_predict / len(predicted)
    return accuracy
