
import tensorflow as tf
from tensorflow import keras as kr
import os
import sys
import tkinter as tk

if len(sys.argv) <= 1 or sys.argv[1] == "--help":
    print("recognize.py  [image]")
    exit(1)

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

img = tf.keras.utils.load_img(
    sys.argv[1], target_size=(img_height, img_width)
    # "./input/kocka_kotevni/Kocka_KOTEVNI_1.jpg", target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

typeAndChance = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))

T.delete(1.0, END)
T.tag_configure("center", justify='center')
T.insert("1.0", typeAndChance)
T.tag_add("center", "1.0", "end")
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
