from __future__ import absolute_import, division, print_function, unicode_literals

''' These are neural network packages '''
import os
import pathlib
import get_meta
import run_model

NAME = ""
SIZE = 0
EPOCHS = 10
BATCH_SIZE = 100
data_path = ""


# Reads a save file
def read_config(file_name):
    global NAME, SIZE, EPOCHS, BATCH_SIZE, data_path
    if ".cfg" not in file_name.lower():
        file_name = file_name + ".cfg"

    if os.path.exists(file_name):
        file = open(file_name, "r")
        f = file.read()
        list = f.split("\n")
        for t in list:
            if "name" in t.lower():
                NAME = t.split("=")[1]
            elif "size" in t.lower():
                SIZE = int(t.split("=")[1])
            elif "epochs" in t.lower():
                EPOCHS = int(t.split("=")[1])
            elif "batch" in t.lower():
                BATCH_SIZE = int(t.split("=")[1])
            elif "path" in t.lower():
                data_path = t.split("=")[1]

        file.close()
        return True
    return False


# Creates a save file
def save_config(file_name):
    global NAME, SIZE, EPOCHS, BATCH_SIZE, data_path
    if ".cfg" not in file_name.lower():
        file_name = file_name + ".cfg"

    if not os.path.exists(file_name):
        file = open(file_name, "w")
        file.write("NAME=" + str(NAME) +
                   "\nSIZE=" + str(SIZE) +
                   "\nEPOCHS=" + str(EPOCHS) +
                   "\nBATCH=" + str(BATCH_SIZE) +
                   "\nPATH=" + str(data_path))
        file.close()
        return True
    return False


def main():
    global NAME, SIZE, EPOCHS, BATCH_SIZE, data_path

    save_name = input("Please Enter a Save name: \n")
    loaded = read_config(save_name)

    if not loaded:
        NAME = save_name
        data_path = pathlib.Path(input("Enter the path of the dataset:\n"))
        SIZE = get_meta.get_img_meta(data_path)
        EPOCHS = input("Enter amount of EPOCHS:\n")
        BATCH_SIZE = input("Enter the BATCH SIZE:\n")

    run_model.run(NAME, SIZE, EPOCHS, BATCH_SIZE, data_path)
    save_config(save_name)


main()
input("Press enter to exit!")
