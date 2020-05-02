import tensorflow as tf
from tensorflow import keras
import os
import pathlib
import numpy as np

test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

NAME = ""
IMG_WIDTH = 0
IMG_HEIGHT = 0
BATCH_SIZE = 50
training_path = pathlib.Path("./dataset/Training")
testing_path = pathlib.Path("./dataset/Testing")
validation_path = pathlib.Path("./dataset/Validation")
CLASS_NAMES = os.listdir(training_path)


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
        return True
    return False


def getDataSetLabels():
    train_list = tf.data.Dataset.list_files(str(training_path / '*/*'))
    labels = []

    for f in train_list:
        label = tf.strings.split(f, "\\")
        label = label[-2]
        labels.append(label)



    '''
    for i, l in labeled_ds:
        print(i)
        keras.preprocessing.image.save_img("p.png", i)
        break
    '''
    return labels


def start():
    global IMG_WIDTH
    global IMG_HEIGHT
    global NAME

    save_name = input("Please Enter a Save name: \n")
    loaded = read_config(save_name)

    if loaded:
        model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu',
                                input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(2, activation='sigmoid')
        ])

        if os.path.exists('./checkpoints/' + NAME + "/model.index"):
            model.load_weights('./checkpoints/' + NAME + '/model').expect_partial()
            print("Model loaded")
        else:
            print("Model doesnt exist")
            exit(0)

        choice = int(input("Would you like to validate the model against a single specific image or a random selection of images?\n1. Specific Image\n2. Random Images\n3. Exit\n\n"))

        if choice == 1:

            file = input("Enter the path of the image:\n")
            img = keras.preprocessing.image.load_img(file, target_size=(IMG_WIDTH, IMG_HEIGHT), color_mode='grayscale')
            img = keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            tf_model_predictions = model.predict(img)

            print("%2s  %10s" % ("#", "Predicted"))

            for x in tf_model_predictions:

                x1 = float(x[0])
                x2 = float(x[1])
                predLB = "Parasitized"
                if x2 > x1:
                    predLB = "Uninfected"


                i = 1
                print("%2i  %12s" % (i, predLB))


        elif choice == 2:
            set_used = int(input("Would you like to use the Test set or the untouched Validation set?\n1. Test\n2. Validation\n\n"))
            true_path = testing_path
            set = ""

            if set_used == 1:
                true_path = testing_path;
                set = "Testing"
            else:
                true_path = validation_path;
                set = "Validation"

            test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                     directory=true_path,
                                                                     target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                     color_mode='grayscale',
                                                                     class_mode='binary',
                                                                     shuffle=True)
            test_image_batch, test_label_batch = next(iter(test_data_gen))

            '''  THIS IS ALL FOR VALIDATING THAT THE VALIDATION SET IS RANDOM AND SHUFFLES CORRECTLY
    
            second_test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                     directory=testing_path,
                                                                     target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                     color_mode='grayscale',
                                                                     class_mode='binary',
                                                                     shuffle=True)
    
            s_test_image_batch, s_test_label_batch = next(iter(second_test_data_gen))
    
            keras.preprocessing.image.save_img("A.png", test_image_batch[0])
            keras.preprocessing.image.save_img("B.png", s_test_image_batch[0])
            '''

            true_label_ids = np.argmax(test_label_batch, axis=-1)
            print(set + " batch shape:", test_image_batch.shape)

            tf_model_predictions = model.predict(test_image_batch)
            print("Prediction results shape:", tf_model_predictions.shape)
            # >> Prediction results shape: (32, 5)

            print("%2s  %10s  %10s  %10s %10s" % ("#", "Parasitized", "Uninfected", "Predicted", "Actual"))
            i = 0
            correct = 0
            total = 0

            for x in tf_model_predictions:
                trueLB = "Parasitized"
                if test_label_batch[i] == 1:
                    trueLB = "Uninfected"

                x1 = float(x[0])
                x2 = float(x[1])
                predLB = "Parasitized"
                if x2 > x1:
                    predLB = "Uninfected"

                if predLB == trueLB:
                    correct += 1
                total += 1

                print("%2i    %2.6f    %2.6f  %12s  %12s" % (i, x1, x2, predLB, trueLB))
                i += 1

            print("correct: " + str(correct))
            print("incorrect: " + str(total - correct))
            print(str(total) + " images predicted with an accuracy of " + str((correct / total) * 100) + "%")

        else:
            exit(1)


start()
input("Press enter to exit!")
