import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

x = ""

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/getInfo', methods=['POST'])
def getInfo():
    if request.method == 'POST':
        file = request.files["file"]

        if file:
            filePath = os.path.join("./static", secure_filename(file.filename))
            file.save(filePath)

            data_dir = pathlib.Path("./input")

            batch_size = 32
            img_height = 180
            img_width = 180

            train_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(img_height, img_width),
                batch_size=batch_size)
            class_names = train_ds.class_names
            print(class_names)

            #################################################

            val_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(img_height, img_width),
                batch_size=batch_size)

            AUTOTUNE = tf.data.AUTOTUNE

            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

            normalization_layer = layers.Rescaling(1./255)

            normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
            image_batch, labels_batch = next(iter(normalized_ds))
            first_image = image_batch[0]
            # Notice the pixel values are now in `[0,1]`.
            print(np.min(first_image), np.max(first_image))

            num_classes = len(class_names)

            data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal",
                                input_shape=(img_height,
                                            img_width,
                                            3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
            )

            model = Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
            ])

            model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

            model.summary()

            epochs = 15
            history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
            )

            ##################################

            model.save("./saved/testModel3")
            model = keras.models.load_model('./saved/testModel3')

            img = tf.keras.utils.load_img(
                filePath, target_size=(img_height, img_width)
            )

            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            typeAndChance = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))

            return typeAndChance

    return "cs"

app.run(debug=True, host='0.0.0.0', port=8080)