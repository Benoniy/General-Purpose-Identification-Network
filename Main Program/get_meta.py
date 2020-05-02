import os
from PIL import Image


def get_img_meta(path_base):
    count = 0
    size = 0
    big = ""
    path_list = [path_base + "/training", path_base + "/testing", path_base + "/validation"]
    CLASS_NAMES = os.listdir(path_list[0])

    for p in path_list:
        for c in CLASS_NAMES:
            a = os.listdir(str(p) + "\\" + c)
            for i in a:
                img = Image.open(str(p) + "\\" + c + "\\" + i)
                w = img.size[0]
                h = img.size[1]
                if w > size:
                    size = w
                    big = str(p) + "\\" + c + "\\" + i
                if h > size:
                    size = h
                    big = str(p) + "\\" + c + "\\" + i
                count = count + 1

    print(big)
    print(str(count) + " images found")
    print("Size set to " + str(size))
    return size
