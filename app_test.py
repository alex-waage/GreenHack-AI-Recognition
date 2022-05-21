# def openfn():
#     filename = filedialog.askopenfilename(title='open')
#     return filename
# def open_img():
    # x = openfn()
    # img = Image.open(x)
    # img = img.resize((250, 250), Image.ANTIALIAS)
    # img = ImageTk.PhotoImage(img)
    # panel = Label(root, image=img)
    # panel.image = img
    # panel.pack()

    # T.delete(1.0, END)
    # T.tag_configure("center", justify='center')
    # T.insert("1.0", "Please Wait.")
    # T.tag_add("center", "1.0", "end")

    # # root.update()
    
    # data_dir = pathlib.Path("./input")

    # # firstType = list(data_dir.glob('delta_kotevni/*'))
    # # PIL.Image.open(str(firstType[0]))
    # # PIL.Image.open(str(firstType[1]))

    # # secondType = list(data_dir.glob('kocka_nosny/*'))
    # # PIL.Image.open(str(secondType[0]))
    # # PIL.Image.open(str(secondType[1]))

    # batch_size = 32
    # img_height = 180
    # img_width = 180

    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=0.2,
    #     subset="training",
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)

    # val_ds = tf.keras.utils.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=0.2,
    #     subset="validation",
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)

    # class_names = train_ds.class_names
    # print(class_names)

    # AUTOTUNE = tf.data.AUTOTUNE

    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # normalization_layer = layers.Rescaling(1./255)

    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_ds))
    # first_image = image_batch[0]
    # # Notice the pixel values are now in `[0,1]`.
    # print(np.min(first_image), np.max(first_image))

    # num_classes = len(class_names)

    # data_augmentation = keras.Sequential(
    # [
    #     layers.RandomFlip("horizontal",
    #                     input_shape=(img_height,
    #                                 img_width,
    #                                 3)),
    #     layers.RandomRotation(0.1),
    #     layers.RandomZoom(0.1),
    # ]
    # )

    # model = Sequential([
    # data_augmentation,
    # layers.Rescaling(1./255),
    # layers.Conv2D(16, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Conv2D(32, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Conv2D(64, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Dropout(0.2),
    # layers.Flatten(),
    # layers.Dense(128, activation='relu'),
    # layers.Dense(num_classes)
    # ])

    # model.compile(optimizer='adam',
    #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #             metrics=['accuracy'])

    # model.summary()

    # epochs = 15
    # history = model.fit(
    # train_ds,
    # validation_data=val_ds,
    # epochs=epochs
    # )

    # ###########################

    # # model = Sequential()
    # # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)))
    # # model.add(layers.MaxPooling2D((2, 2)))
    # # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # # model.add(layers.MaxPooling2D((2, 2)))
    # # model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # # model.summary()

    # # model.add(layers.Flatten())
    # # model.add(layers.Dense(64, activation='relu'))
    # # model.add(layers.Dense(10))

    # # model.summary()

    # # model.compile(optimizer='adam',
    # #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # #             metrics=['accuracy'])

    # # epochs = 15
    # # history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


    # ##################################

    # # model.save("./saved/testModel")
    # # model = keras.models.load_model('./saved/testModel')

    # img = tf.keras.utils.load_img(
    #     "./input_test_prediction/test_1.jpg", target_size=(img_height, img_width)
    #     # "./input/kocka_kotevni/Kocka_KOTEVNI_1.jpg", target_size=(img_height, img_width)
    # )
    # img_array = tf.keras.utils.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0) # Create a batch

    # predictions = model.predict(img_array)
    # score = tf.nn.softmax(predictions[0])

    # typeAndChance = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))

    # T.delete(1.0, END)
    # T.tag_configure("center", justify='center')
    # T.insert("1.0", typeAndChance)
    # T.tag_add("center", "1.0", "end")
    # print(
    #     "This image most likely belongs to {} with a {:.2f} percent confidence."
    #     .format(class_names[np.argmax(score)], 100 * np.max(score))
    # )


# btn = Button(root, text='Open image', command=open_img).pack()
# print(x)
# T = Text(root, height=2, width=50)
# T.tag_configure("center", justify='center')
# T.insert("1.0", "Please import image.")
# T.tag_add("center", "1.0", "end")
# T.pack()

# root.mainloop()

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

            # val_ds = tf.keras.utils.image_dataset_from_directory(
            #     data_dir,
            #     validation_split=0.2,
            #     subset="validation",
            #     seed=123,
            #     image_size=(img_height, img_width),
            #     batch_size=batch_size)

            # class_names = train_ds.class_names
            # print(class_names)

            # AUTOTUNE = tf.data.AUTOTUNE

            # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
            # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

            # normalization_layer = layers.Rescaling(1./255)

            # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
            # image_batch, labels_batch = next(iter(normalized_ds))
            # first_image = image_batch[0]
            # # Notice the pixel values are now in `[0,1]`.
            # print(np.min(first_image), np.max(first_image))

            # num_classes = len(class_names)

            # data_augmentation = keras.Sequential(
            # [
            #     layers.RandomFlip("horizontal",
            #                     input_shape=(img_height,
            #                                 img_width,
            #                                 3)),
            #     layers.RandomRotation(0.1),
            #     layers.RandomZoom(0.1),
            # ]
            # )

            # model = Sequential([
            # data_augmentation,
            # layers.Rescaling(1./255),
            # layers.Conv2D(16, 3, padding='same', activation='relu'),
            # layers.MaxPooling2D(),
            # layers.Conv2D(32, 3, padding='same', activation='relu'),
            # layers.MaxPooling2D(),
            # layers.Conv2D(64, 3, padding='same', activation='relu'),
            # layers.MaxPooling2D(),
            # layers.Dropout(0.2),
            # layers.Flatten(),
            # layers.Dense(128, activation='relu'),
            # layers.Dense(num_classes)
            # ])

            # model.compile(optimizer='adam',
            #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            #             metrics=['accuracy'])

            # model.summary()

            # epochs = 15
            # history = model.fit(
            # train_ds,
            # validation_data=val_ds,
            # epochs=epochs
            # )

            ##################################

            # model.save("./saved/testModel")
            model = keras.models.load_model('./saved/testModel')

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