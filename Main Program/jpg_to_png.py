from PIL import Image
import os

path = input("Please enter the dir of the images you want to convert:\n")
output = path.split("\\")
output.pop(len(output) - 1)
output.append("pngConverted")

outpath = ""
for o in output:
    outpath += o + "\\"


files = os.listdir(path)
if not os.path.exists(outpath):
    os.mkdir(outpath)

for f in files:
    img = Image.open(path + "\\" + f)
    outf = f.replace(".jpg", ".png")
    img.save(outpath + outf)
