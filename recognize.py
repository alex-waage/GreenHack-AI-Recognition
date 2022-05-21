
import tensorflow as tf
from tensorflow import keras as kr
import os
import sys
import tkinter as tk

usegui = len(sys.argv) <= 1
def crash(e, code = 5):
    if usegui:
        greeting = tk.Label(text="Hello, Tkinter")
        tk.pack()
    else:
        print(e)
    exit(code)

#Load the model
mpath = os.environ.get("MODEL_PATH") or "models/default";

try:
    model = tf.keras.models.load_model(mpath);
    #model = tf.savedmodel.load(mpath);
except:
    crash("ERROR: Model could not be loaded from \"%s\"" % (mpath));

crash("ERROR: Model could not be loaded")

fashion_mnist = tf.keras.datasets.fashion_mnist

#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

