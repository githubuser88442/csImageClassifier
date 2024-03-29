import os, os.path
from pathlib import Path
import time

import mss
import mss.tools
from pynput.keyboard import Key, KeyCode, Listener

"""
Tool to get and label images for training data.
Waits for keyinput, "o" for to label image as shoot (label), "p" to label image as background, "x" to exit.
"""
DATA_PATH = Path('data/extra/200')
FILE_COUNT_SHOOT = len([name for name in os.listdir(DATA_PATH/'shoot')])
FILE_COUNT_BG = len([name for name in os.listdir(DATA_PATH/'bg')])

def get_image(count=0, label='img', dir=''):
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
        # Save it!
        mss.tools.to_png(img.rgb, img.size, output=DATA_PATH/'{}/{}.{}.png'.format(dir, count, label))


print('Shoot, bg count:', FILE_COUNT_SHOOT, FILE_COUNT_BG)

RECORDING_SHOOT = False # Ter class
RECORDING_BG = False # Bg class

print('Waiting for input..')
def on_press(key):
    pass

def on_release(key):
    # Global to update the file count without having to recount
    global FILE_COUNT_BG
    global FILE_COUNT_SHOOT

    if key == KeyCode(char='o'):
        print('Shoot image', FILE_COUNT_SHOOT)
        for i in range(10):
            FILE_COUNT_SHOOT += 1
            get_image(count=FILE_COUNT_SHOOT, label='ter', dir='shoot')
            time.sleep(0.0)

    if key == KeyCode(char='p'):
        print('Bg image', FILE_COUNT_BG)
        for i in range(10):
            FILE_COUNT_BG += 1
            get_image(count=FILE_COUNT_BG, label='bg', dir='bg')
            time.sleep(0.02)


    if key == KeyCode(char='x'):
        # Stop listener
        print('Exiting..')
        return False

# Collect events until released
# with Listener(on_press=on_press, on_release=lambda key: on_release(key,opt)) as listener:
with Listener(on_release=on_release) as listener:
    listener.join()