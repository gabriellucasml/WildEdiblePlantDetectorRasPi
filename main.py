# from picamera import PiCamera
import os

import pandas as pd
import requests
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from pyfirmata2 import Arduino, util, STRING_DATA, INPUT


# def takPicture():
#   camera = PiCamera()
#   camera.resolution(256,256)
#   camera.start_preview()
#   sleep(5)
#   camera.capture(os.getcwd()+'/capturedImages/img'+str(len(os.listdir(os.getcwd()+'/capturedImages/')))+'.png')
#   return os.getcwd()+'/capturedImages/img'+str(len(os.listdir(os.getcwd()+'/capturedImages/')))+'.png'

def msg(text):
    if text:
        board.send_sysex(STRING_DATA, util.str_to_two_byte_iter(text))
    else:
        board.send_sysex(STRING_DATA, util.str_to_two_byte_iter(' '))


def load(imgpath, model, class_names):
    msg('Predicting...')
    img = tf.keras.utils.load_img(imgpath)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    msg('Predicted')
    screens = [
        ["This image most", "likely belongs to"],
        ["{} with a ".format(class_names[np.argmax(score)][0]), "{:.2f}% confidence.".format(100 * np.max(score))]
    ]
    return screens


model = keras.models.load_model(os.getcwd() + '\wild_edible_plants_detector')
board = Arduino('COM5')
img = r'D:\Files\UFRN\IA\WildEdiblePlantDetector\dataset\processed\testing\Calendula\Calendula399.png'
class_names = pd.read_csv(os.getcwd() + '\wild_edible_plants_detector\class_names.csv').values.tolist()
class_names.insert(0, ['Alfafa'])

it = util.Iterator(board)
it.start()

button = board.digital[13]
button.mode = INPUT

screens = load(img, model, class_names)
start_time = time.time()
while True:
    time_now = time.time()
    if button.read() == 1:
        # takePicture()
        # img = os.getcwd()+'/capturedImages/img'+str(len(os.listdir(os.getcwd()+'/capturedImages/')))+'.png'
    if (time_now - start_time) > 1800.0:
        screens = load(img, model, class_names)
        start_time = time.time()
    for screen in screens:
        msg(screen[0])
        time.sleep(0.01)
        msg(screen[1])
        time.sleep(2)
