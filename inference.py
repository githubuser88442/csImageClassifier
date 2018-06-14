import os, os.path
from pathlib import Path
import numpy as np
import time

from pynput.keyboard import Key, KeyCode, Listener
import mss
import mss.tools

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

import torch
from torch.autograd.variable import Variable
from PIL import Image

from matplotlib import pyplot as plt


PATH = Path('data/150/')
#MODEL = torch.load(PATH/"models/torch_model_v1").eval()
# print(MODEL)
sz = 224
arch=resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
print('Loading..')
learn = ConvLearner.pretrained(arch, data, precompute=False)
print('ok')
learn.load('csgo_all')
COUNT = 0

def return_image(count=0, label='img', dir=''):
    with mss.mss() as sct:
        # Use the 1st monitor
        monitor = sct.monitors[1]

        # Capture a bbox using percent values
        left = monitor['left'] + monitor['width'] * 45 // 100 - 4 # 50% from the left
        top = monitor['top'] + monitor['height'] * 41 // 100 - 3  # 50% from the top minus some px
        right = left + 200  # 200px width
        lower = top + 200  # 200px height
        bbox = (left, top, right, lower)
        img = sct.grab(bbox)
        # if COUNT % 20 == 0:
        #     mss.tools.to_png(img.rgb, img.size, output=PATH/'{}/{}.png'.format("random", COUNT))
        img = np.array(Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX'), dtype=np.float32)
        return img

def predict(img):
    # Preprocess image
    to_predict = np.expand_dims(img, 0)
    sh = to_predict.shape
    to_predict = to_predict.reshape(sh[0], sh[-1], sh[1], sh[2])

    log_pred = learn.predict_array(to_predict)
    #log_pred = MODEL(Variable(torch.from_numpy(to_predict)).cuda()).cpu().data.numpy()
    pred = np.argmax(log_pred, axis=1)  # from log probabilities to 0 or 1

    #plt.imshow(img.astype(np.int32), interpolation='nearest')
    #plt.show()
    return pred

def test_on_image(img_name):
    """ very quick method to test on a single image in /random/ folder """
    test_img = np.array(Image.open('C:/Users/Ranet/Documents/Machine Learning/games/cs_trig/data/150/random/' + img_name), dtype=np.float32)
    predict(test_img)

# log_preds,y = learn.TTA()
# probs = np.mean(np.exp(log_preds),0)
# print(accuracy_np(probs, y))

while True:
    #COUNT += 1
    pred = predict(return_image())
    print(pred, data.classes[pred[0]])
    time.sleep(0.2)
