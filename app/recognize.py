
import tensorflow as tf
from tensorflow import keras as kr
import os
import sys
import tkinter as tk

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np

batch_size = 32
img_height = 180
img_width = 180

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename


if len(sys.argv) > 1 and sys.argv[1] == "--help":
    print("recognize.py  [image]")
    exit(1)

usegui = len(sys.argv) <= 1
try:
    impath = sys.argv[1]
except IndexError:
    impath = openfn() if usegui else input("Image name")
def crash(e, code = 5):
    if usegui:
        print(e)
        greeting = tk.Label(text="Hello, Tkinter")
        tk.Pack()
    else:
        print(e)
    exit(code)

#Load the model
mpath = os.environ.get("MODEL_PATH") or "models/default";

try:
    model = tf.keras.models.load_model(mpath);
    class_names = open(mpath + "/class_names.list").read().splitlines()
    #model = tf.savedmodel.load(mpath);
except BaseException as e:
    crash("ERROR: Model could not be loaded from \"%s\"\nException: %s" % (mpath, str(e)));

img = tf.keras.utils.load_img(
    impath, target_size=(img_height, img_width)
    # "./input/kocka_kotevni/Kocka_KOTEVNI_1.jpg", target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

typeAndChance = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))

#tk.delete(1.0, END)
#tk.tag_configure("center", justify='center')
#tk.insert("1.0", typeAndChance)
#tk.tag_add("center", "1.0", "end")
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
