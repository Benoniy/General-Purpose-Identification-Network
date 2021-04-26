import os
import pathlib
import get_meta
import train_model
from save_load_model import *

NAME = ""
SIZE = 0
EPOCHS = 10
BATCH_SIZE = 100
data_path = ""


def main():
    global NAME, SIZE, EPOCHS, BATCH_SIZE, data_path

    save_name = input("Please Enter a Save name: \n")
    model = read_config(save_name)

    if not model:
        name = save_name
        data_path = pathlib.Path(input("Enter the path of the dataset:\n"))
        size = get_meta.get_img_meta(data_path)
        epochs = input("Enter amount of EPOCHS:\n")
        batch_size = input("Enter the BATCH SIZE:\n")
        model = Model(name, size, epochs, batch_size, data_path)

    train_model.run(model.NAME, model.SIZE, model.EPOCHS, model.BATCH_SIZE, model.data_path)

    save_config(save_name, model)


main()
input("Press enter to exit!")
