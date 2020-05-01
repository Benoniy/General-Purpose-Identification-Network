from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow import keras
import os
import pathlib
import get_meta


# Constant variables
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_WIDTH = 0
IMG_HEIGHT = 0
NAME = ""

EPOCHS = 10
BATCH_SIZE = 100
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
training_path = pathlib.Path("./dataset/Training")
testing_path = pathlib.Path("./dataset/Testing")
validation_path = pathlib.Path("./dataset/Validation")
path_list = [training_path, testing_path, validation_path]
CLASS_NAMES = os.listdir(training_path)
test_count = 1
TRN_IMAGES = []
TRN_LABELS = []


def read_config(file_name):
    global IMG_WIDTH
    global IMG_HEIGHT
    global NAME
    if ".cfg" not in file_name.lower():
        file_name = file_name + ".cfg"

    if os.path.exists(file_name):
        file = open(file_name, "r")
        f = file.read()
        list = f.split("\n")
        NAME = list[0].split("=")[1]
        IMG_HEIGHT = int(list[1].split("=")[1])
        IMG_WIDTH = IMG_HEIGHT
        file.close()
        return True
    return False


def save_config(file_name):
    global IMG_WIDTH
    global IMG_HEIGHT
    global NAME
    if ".cfg" not in file_name.lower():
        file_name = file_name + ".cfg"

    if not os.path.exists(file_name):
        file = open(file_name, "w")
        file.write("NAME=" + str(NAME) + "\nSIZE=" + str(IMG_HEIGHT))
        file.close()
        return True
    return False


def get_label(file_path):
    pt = tf.strings.split(file_path, "\\")
    return pt[-2] == CLASS_NAMES


# This decodes each image into a tf.image object and performs pre-processing
def decode_img(img, file_path):

    img = tf.image.decode_png(img, channels=1)


    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img, file_path)
    return img, label


# This controls the creation of the dataset
def getDataSet():
    train_list = tf.data.Dataset.list_files(str(training_path / '*/*'))

    labeled_ds = train_list.map(process_path, num_parallel_calls=AUTOTUNE)


    '''
    for i, l in labeled_ds:
        print(i)
        keras.preprocessing.image.save_img("p.png", i.numpy())
        break
    '''
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
    global IMG_WIDTH
    global IMG_HEIGHT
    global NAME

    save_name = input("Please Enter a Save name: \n")
    loaded = read_config(save_name)

    if not loaded:
        NAME = save_name
        IMG_HEIGHT = get_meta.get_img_meta(path_list, CLASS_NAMES)
        IMG_WIDTH = IMG_HEIGHT

    dataset = getDataSet()
    train_ds = prepare_for_training(dataset)
    train_image_batch, train_label_batch = next(iter(train_ds))

    # Simple sequential
    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(len(CLASS_NAMES), activation='sigmoid')
    ])

    if os.path.exists('./checkpoints/' + NAME + "/model.index"):
        model.load_weights('./checkpoints/' + NAME + "/model")
        print("Model loaded")

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(train_ds, steps_per_epoch=BATCH_SIZE, epochs=EPOCHS)
    model.save_weights('./checkpoints/' + NAME + "/model")

    save_config(save_name)


train()
input("Press enter to exit!")
