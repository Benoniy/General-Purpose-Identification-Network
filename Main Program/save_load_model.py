import os
from model import Model


def save_config(file_name, model):
    """ Creates a save file that can be used to access the model again """
    if ".cfg" not in file_name.lower():
        file_name = file_name + ".cfg"

    if not os.path.exists(file_name):
        file = open(file_name, "w")
        file.write("NAME=" + str(model.NAME) +
                   "\nSIZE=" + str(model.SIZE) +
                   "\nEPOCHS=" + str(model.EPOCHS) +
                   "\nBATCH=" + str(model.BATCH_SIZE) +
                   "\nPATH=" + str(model.data_path))
        file.close()
        return True
    return False


def read_config(file_name):
    """ Reads a save file which references a model """
    name = ""
    size = ""
    epochs = ""
    batch_size = ""
    data_path = ""
    if ".cfg" not in file_name.lower():
        file_name = file_name + ".cfg"

    if os.path.exists(file_name):
        file = open(file_name, "r")
        f = file.read()
        list = f.split("\n")
        for t in list:
            if "name" in t.lower():
                name = t.split("=")[1]
            elif "size" in t.lower():
                size = int(t.split("=")[1])
            elif "epochs" in t.lower():
                epochs = int(t.split("=")[1])
            elif "batch" in t.lower():
                batch_size = int(t.split("=")[1])
            elif "path" in t.lower():
                data_path = t.split("=")[1]

        file.close()
        return Model(name, size, epochs, batch_size, data_path)
    return False
