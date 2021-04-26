import tensorflow as tf
from tensorflow import keras
import os
import pathlib
import numpy as np
import shutil

test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)


def run(name, size, batch_size, data_path, validation, specific_path, single):
    shutil.rmtree(".\\cache")
    os.mkdir(".\\cache")

    training_path = pathlib.Path(data_path + "/training")
    testing_path = pathlib.Path(data_path + "/testing")
    validation_path = pathlib.Path(data_path + "/validation")
    class_names = os.listdir(training_path)

    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu',
                            input_shape=(size, size, 1)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])

    if os.path.exists('./checkpoints/' + name + "/model.index"):
        model.load_weights('./checkpoints/' + name + '/model').expect_partial()
        print("Model loaded")
    else:
        print("Model doesnt exist")
        exit(0)



    true_path = testing_path
    set = ""

    if not validation:
        true_path = testing_path
        set = "Testing"
    else:
        true_path = validation_path
        set = "Validation"

    if single:
        path_split = specific_path.split("\\")
        for c in class_names:
            os.mkdir(".\\cache\\" + c)
        shutil.copy(specific_path, ".\\cache\\" + "Uninfected" + "\\" + path_split[-1])
        true_path = ".\\cache\\"
        batch_size = 1

    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                             directory=true_path,
                                                             target_size=(size, size),
                                                             color_mode='grayscale',
                                                             class_mode='binary',
                                                             shuffle=True)

    test_image_batch, test_label_batch = next(iter(test_data_gen))

    true_label_ids = np.argmax(test_label_batch, axis=-1)
    print("\n" + set + " batch shape:", test_image_batch.shape)

    tf_model_predictions = model.predict(test_image_batch)
    print("Prediction results shape:", tf_model_predictions.shape, "\n")

    stringFormat = "%3s   "
    titleConfidence = 0
    titleConfidenceText = ""
    for c in class_names:
        titleConfidence += 10
        titleConfidenceText += str(c) + "  "




    if single:
        stringFormat += "%" + str(titleConfidence) + "s    %10s"
        print(stringFormat % ("#", titleConfidenceText, "Predicted"))
    else:
        stringFormat += "%" + str(titleConfidence) + "s    %10s    %10s"
        print(stringFormat % ("#", titleConfidenceText, "Predicted", "Actual"))

    i = 0
    correct = 0
    total = 0

    for x in tf_model_predictions:
        trueLB = class_names[int(test_label_batch[i])]
        predLB = ""
        high = 0
        count = 0
        confid = []

        for c in x:
            confid.append(c)
            if c > high:
                high = c
                predLB = str(class_names[count])
            count = count + 1

        if predLB == trueLB:
            correct += 1
        total += 1

        stringFormat = "%3i"
        a = 0
        confidOut = "  "
        for c in confid:
            a += 11
            confidOut += "%2.6f    " % c

        if single:
            stringFormat += "    %" + str(a) + "s" + "%12s"
            print(stringFormat % (i + 1, confidOut, predLB))
        else:
            stringFormat += "    %" + str(a) + "s" + "%12s  %12s"
            print(stringFormat % (i + 1, confidOut, predLB, trueLB))

        i += 1

    if not single:
        print("correct: " + str(correct))
        print("incorrect: " + str(total - correct))
        print(str(total) + " images predicted with an accuracy of " + str((correct / total) * 100) + "%")

    shutil.rmtree(".\\cache")
    os.mkdir(".\\cache")

