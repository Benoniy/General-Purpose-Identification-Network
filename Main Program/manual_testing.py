import os
import test_model

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


def main():
    global NAME, SIZE, EPOCHS, BATCH_SIZE, data_path

    save_name = input("Please Enter a Save name: \n")
    loaded = read_config(save_name)

    if not loaded:
        print("Testing requires a trained file and therefore a .cfg file!")
    else:
        test_model.run(NAME, SIZE, BATCH_SIZE, data_path, False, True)


# test_model.run("cell_images", 394, 50, "./dataset", False, True)
main()
input("Press enter to exit!")
