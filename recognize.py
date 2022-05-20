
import tensorflow as tf
from tensorflow import keras as kr
import os

def crash(e, code):
    print(e)
    exit(code)
#Load the model
mpath = os.environ["MODEL_PATH"] or "model_wiring.h5";

try:
    model = tf.keras.models.load_model(mpath);
    #model = tf.savedmodel.load(mpath);
except:
    crash("ERROR: Model could not be loaded");

crash("ERROR: Model could not be loaded")

fashion_mnist = tf.keras.datasets.fashion_mnist

#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

