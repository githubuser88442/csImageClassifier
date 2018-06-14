from PIL import Image
import os.path, sys
"""quick tool to crop images
"""

path_val_shoot = "C:/Users/Ranet/Documents/Machine Learning/games/cs_trig/data/extra/200/shoot"
path_val_bg = "C:/Users/Ranet/Documents/Machine Learning/games/cs_trig/data/extra/200/bg"
# path_train_shoot = "C:/Users/Ranet/Documents/Machine Learning/games/cs_trig/data/crosshair/valid/shoot"
# path_train_bg = "C:/Users/Ranet/Documents/Machine Learning/games/cs_trig/data/crosshair/valid/bg"

def crop(path):
    dirs = os.listdir(path)
    for item in dirs:
        fullpath = os.path.join(path,item)
        if os.path.isfile(fullpath):
            img = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            # imCrop = img.crop((62, 10, 130, 130))
            # imCrop.save(f + ".png", "PNG", quality=10)
            img.save(os.path.join(path, "name/"+ item + "_200" + e), "PNG", quality=10)

crop(path_val_shoot)
crop(path_val_bg)
# crop(path_train_shoot)
# crop(path_train_bg)