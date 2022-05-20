
import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt



batch_size = 32
img_height = 1280
img_width = 960



interactive = os.environ.get("NO_INTERACT") == None
#plt.rcParams['axes.facecolor'] = 'black'
plt.style.use('dark_background')

df_modelpath = "./Pictures_sorted/";

print("Getting model path   -- from PIC_PATH environ, or from  \"" + df_modelpath + "\" by default")
fld = os.environ.get("PIC_PATH") or df_modelpath;
print("Training model")

dataset= tf.keras.utils.image_dataset_from_directory(
    fld,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
);

#Validation dataset
validation_ds = tf.keras.utils.image_dataset_from_directory(
    fld,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


class_nms = dataset.class_names
print(f"class_nms: {class_nms}")
class_cnt = len(class_nms);

#plt.gca().set_facecolor([0.0, 0.2, 0.0])

if interactive:
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(class_cnt):
            print(f"Showing image {i}")
            ax = plt.subplot(3, 3, i + 1)
            plt.title(class_nms[labels[i]])
            plt.axis("off")
            
            plt.imshow(images[i].numpy().astype("uint8"))
        
plt.show()

model = kr.Sequential([
    layers.Rescaling(1./1024, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    #layers.Dense(128, activation='relu'),
    #layers.Dense(class_cnt)
])

print("Compiling model...")
model.compile(optimizer='adam',
    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss='mean_squared_error'
    #metrics=['accuracy']
)
print("Training model...")

history = model.fit(
    dataset,
    validation_data=None,
    epochs=int(os.environ.get("TRAIN_EPOCHS") or 9),
    #input_shape=(img_height, img_width, 3)
    )
#history = model.fit(
    #dataset,
    #validation_data=val_ds,
    #epochs=epochs
    #)
data_augmentation = kr.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
  ]
)



print("\nModel summary:\n%s\n" % (model.summary))

#class_names[i]
print("Successfully terminated")
