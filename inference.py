import os, os.path
from pathlib import Path
import numpy as np
import time
from PIL import Image
from matplotlib import pyplot as plt

import mss
import mss.tools
from pynput.keyboard import Key, KeyCode, Listener

from fastai.model import *
from torchvision.transforms import Normalize, Compose, CenterCrop, Resize, ToTensor


PATH = Path('data/150/')
# PyTorch model
print('Loading..')
MODEL = torch.load(PATH/"models/torch_model_v1").eval()
CLASSES = ['bg', 'shoot']
print('Model loaded!')

def return_image(**kwargs):
    """Take screenshot and return the image.
    **kwargs for debugging: save images to a folder - pass in save_img=True, count=(int), path"""
    with mss.mss() as sct:
        # Use the 1st monitor
        monitor = sct.monitors[1]

        # Capture a bbox using percent values
        left = monitor['left'] + monitor['width'] * 45 // 100 - 4 # 45% from the left minus some px
        top = monitor['top'] + monitor['height'] * 41 // 100 - 3  # 45% from the top
        right = left + 200  # 200px width
        lower = top + 200  # 200px height
        bbox = (left, top, right, lower)
        img = sct.grab(bbox)

        # Save images to a folder for debugging, pass in save_img, count, path
        if 'save_img' in kwargs and kwargs['save_img']:
            if kwargs['count'] % 20 == 0:
                mss.tools.to_png(img.rgb, img.size, output=kwargs['path']/'{}/{}.png'.format("random", kwargs['count']))

        img = Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')
        return img

def predict_on_img(img):
    """Predict on a single PIL Image/np array"""

    # Normalize images
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = Compose([Resize(256), CenterCrop(224), ToTensor(), normalize])

    # Add a dim and preprocess
    img_tensor = preprocess(img).unsqueeze_(0)
    img_variable = Variable(img_tensor.cuda())

    # Predict
    log_probs = MODEL(img_variable)
    preds = np.argmax(log_probs.cpu().data.numpy(), axis=1)

    return preds

def inference():
    """Gets an img via screenshot and predicts on it in realtime"""

    print('Starting inference!')
    while_count = 0
    run = True
    s_time = time.time()
    while run:
        pred = predict_on_img(return_image(save_img=False, count=while_count, path=PATH))
        #print(pred, CLASSES[pred[0]])

        while_count += 1
        if while_count >= 60:
            run = False
            print('60 frames processed, took {}'.format(time.time() - s_time))

def test_on_image(img_name):
    """ very quick method to test on a single image in /random/ folder """
    test_img = Image.open(PATH/'/random/' + img_name)
    print(predict(test_img))

def test_on_folder(path, class_id):
    """Quick debug tool to test on an entire folder, print accuracy"""
    import glob
    preds = []
    for filename in glob.glob(path + '/*.png'):
        img = np.array(Image.open(filename), dtype=np.float32)
        preds.append(predict(img)[0])

    num_preds = len(preds)
    correct_count = preds.count(class_id)
    print("Num images:", num_preds)
    print("Num correct:", correct_count)
    print("Acc:", correct_count / num_preds)

# test_on_folder(PATH/"valid/shoot", 1)
# test_on_image("valid.png") #200

# Call inference
inference()