from __future__ import absolute_import, division, print_function, unicode_literals
import os

import tensorflow as tf
from tensorflow import keras
import os
import pathlib

# These should be left alone
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
SETUP = False

NAME = ""
SIZE = 0
EPOCHS = 0
BATCH_SIZE = 0

training_path = pathlib.Path("")
testing_path = pathlib.Path("")
validation_path = pathlib.Path("")
path_list = []
CLASS_NAMES = []


def get_label(file_path):
    pt = tf.strings.split(file_path, "\\")
    return pt[-2] == CLASS_NAMES


def decode_img(img, file_path):
    """ This decodes each image into a tf.image object and performs pre-processing """
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [SIZE, SIZE])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img, file_path)
    return img, label


def getDataSet():
    """ This controls the creation of the dataset """
    train_list = tf.data.Dataset.list_files(str(training_path / '*/*'))
    labeled_ds = train_list.map(process_path, num_parallel_calls=AUTOTUNE)
    return labeled_ds


def prepare_for_training(ds, cache=False, shuffle_buffer_size=20668):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def train():
    global SIZE
    global NAME

    if SETUP:
        dataset = getDataSet()
        file_count = sum(len(files) for _, _, files in os.walk(rf"{training_path}"))
        train_ds = prepare_for_training(dataset, shuffle_buffer_size=file_count)
        train_image_batch, train_label_batch = next(iter(train_ds))  # Depreciated

        # Simple sequential Neural network
        model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu',
                                input_shape=(SIZE, SIZE, 1)),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
        ])

        if os.path.exists('./checkpoints/' + NAME + "/model.index"):
            model.load_weights('./checkpoints/' + NAME + "/model")
            print("Model loaded")

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit_generator(train_ds, steps_per_epoch=BATCH_SIZE, epochs=EPOCHS)
        model.save_weights('./checkpoints/' + NAME + "/model")
    else:
        print("Use the run() method first to set required variables!")


def run(name, im_size, epochs, batch_size, data_path):
    global NAME, SIZE, EPOCHS, BATCH_SIZE, training_path, testing_path, validation_path, path_list, CLASS_NAMES, SETUP
    NAME = name
    SIZE = im_size
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    training_path = pathlib.Path(data_path + "/training")
    testing_path = pathlib.Path(data_path + "/testing")
    validation_path = pathlib.Path(data_path + "/validation")
    path_list = [training_path, testing_path, validation_path]
    CLASS_NAMES = os.listdir(training_path)
    SETUP = True

    train()
