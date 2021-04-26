import os
import test_model
from save_load_model import read_config


def test():

    save_name = input("Please Enter a Save name: \n")
    model = read_config(save_name)
    val = False
    single = False
    path = None

    if not model:
        print("Testing requires a trained file and therefore a .cfg file!")
    else:
        option_one = input("Do you want to: \n1)Test a set of random images \n2)Test a specific image\n")
        if "2" in option_one:
            path = input("Please enter the path of the image you want to test:\n")
            single = True
        else:
            path = None
            val = input("Do you want to use the validation set (y/n):\n")
            if "y" in val.lower():
                val = True
            else:
                val = False

        test_model.run(model.NAME, model.SIZE, model.BATCH_SIZE, model.data_path, val, path, single)


if __name__ == "__main__":
    # test_model.run("cell_images", 394, 50, "./dataset", False, True)
    test()
    input("Press enter to exit!")
