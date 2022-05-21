import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import os
import pathlib
import sys


batch_size = 32
img_height = 180
img_width = 180

if len(sys.argv) <= 1 or sys.argv[1] == "--help":
    print("train_model.py  [output path]")
    exit(1)

outfile = sys.argv[1]

x = ""
data_dir = "Pictures_sorted"

#root = Tk()
#root.resizable(width=True, height=True)
#root.eval('tk::PlaceWindow . center')


def train_model():
    #x = openfn()
    #img = Image.open(x)
    #img = img.resize((250, 250), Image.ANTIALIAS)
    #img = ImageTk.PhotoImage(img)
    #panel = Label(root, image=img)
    #panel.image = img
    #panel.pack()

    #T.delete(1.0, END)
    #T.tag_configure("center", justify='center')
    #T.insert("1.0", "Please Wait.")
    #T.tag_add("center", "1.0", "end")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(class_names)

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
    open(outfile + "/class_names.list", "w").write('\n'.join(class_names));

    return model



#if train_model() == True:
    #print("Successfully trained")
#else:
    #print("ERROR: Cannot train model due to unknown error")

train_model().save(outfile)
print("Successfully trained")

# data_dir = pathlib.Path("./input")

# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)

# roses = list(data_dir.glob('delta_kotevni/*'))
# PIL.Image.open(str(roses[0]))

# PIL.Image.open(str(roses[1]))

# tulips = list(data_dir.glob('kocka_nosny/*'))
# PIL.Image.open(str(tulips[0]))

# PIL.Image.open(str(tulips[1]))

# batch_size = 32
# img_height = 180
# img_width = 180

# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# class_names = train_ds.class_names
# print(class_names)

# import matplotlib.pyplot as plt

# # plt.figure(figsize=(10, 10))
# # for images, labels in train_ds.take(1):
# #   for i in range(9):
# #     ax = plt.subplot(3, 3, i + 1)
# #     plt.imshow(images[i].numpy().astype("uint8"))
# #     plt.title(class_names[labels[i]])
# #     plt.axis("off")

# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break

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

# # #############################################################

# # model = Sequential([
# #   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
# #   layers.Conv2D(16, 3, padding='same', activation='relu'),
# #   layers.MaxPooling2D(),
# #   layers.Conv2D(32, 3, padding='same', activation='relu'),
# #   layers.MaxPooling2D(),
# #   layers.Conv2D(64, 3, padding='same', activation='relu'),
# #   layers.MaxPooling2D(),
# #   layers.Flatten(),
# #   layers.Dense(128, activation='relu'),
# #   layers.Dense(num_classes)
# # ])

# # model.compile(optimizer='adam',
# #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
# #               metrics=['accuracy'])

# # model.summary()

# # epochs=10
# # history = model.fit(
# #   train_ds,
# #   validation_data=val_ds,
# #   epochs=epochs
# # )

# # acc = history.history['accuracy']
# # val_acc = history.history['val_accuracy']

# # loss = history.history['loss']
# # val_loss = history.history['val_loss']

# # epochs_range = range(epochs)

# # #############################################################

# # plt.figure(figsize=(8, 8))
# # plt.subplot(1, 2, 1)
# # plt.plot(epochs_range, acc, label='Training Accuracy')
# # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# # plt.legend(loc='lower right')
# # plt.title('Training and Validation Accuracy')

# # plt.subplot(1, 2, 2)
# # plt.plot(epochs_range, loss, label='Training Loss')
# # plt.plot(epochs_range, val_loss, label='Validation Loss')
# # plt.legend(loc='upper right')
# # plt.title('Training and Validation Loss')
# # plt.show()

# data_augmentation = keras.Sequential(
#   [
#     layers.RandomFlip("horizontal",
#                       input_shape=(img_height,
#                                   img_width,
#                                   3)),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#   ]
# )

# # plt.figure(figsize=(10, 10))
# # for images, _ in train_ds.take(1):
# #   for i in range(9):
# #     augmented_images = data_augmentation(images)
# #     ax = plt.subplot(3, 3, i + 1)
# #     plt.imshow(augmented_images[0].numpy().astype("uint8"))
# #     plt.axis("off")

# model = Sequential([
#   data_augmentation,
#   layers.Rescaling(1./255),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Dropout(0.2),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.summary()

# epochs = 15
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# # plt.figure(figsize=(8, 8))
# # plt.subplot(1, 2, 1)
# # plt.plot(epochs_range, acc, label='Training Accuracy')
# # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# # plt.legend(loc='lower right')
# # plt.title('Training and Validation Accuracy')

# # plt.subplot(1, 2, 2)
# # plt.plot(epochs_range, loss, label='Training Loss')
# # plt.plot(epochs_range, val_loss, label='Validation Loss')
# # plt.legend(loc='upper right')
# # plt.title('Training and Validation Loss')
# # plt.show()

# # sunflower_url = "./input_test_prediction/test_1.jpg"
# # sunflower_path = tf.keras.utils.get_file('stozar', origin=sunflower_url)

# img = tf.keras.utils.load_img(
#     "./input_test_prediction/test_1.jpg", target_size=(img_height, img_width)
#     # "./input/kocka_kotevni/Kocka_KOTEVNI_1.jpg", target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
