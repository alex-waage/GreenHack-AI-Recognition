
import tensorflow as tf
from tensorflow import keras as kr
import os
import matplotlib.pyplot as plt

df_modelpath = "./Pictures_sorted/";

print("Getting model path   -- from PIC_PATH environ, or from  \"" + df_modelpath + "\" by default")
fld = os.environ.get("PIC_PATH") or df_modelpath;
print("Training model")
dataset= tf.keras.utils.image_dataset_from_directory(fld);

class_nms = dataset.class_names
print(f"class_nms: {class_nms}")
class_cnt = len(class_nms);

plt.figure(figsize=(10, 10))
for images, labels in dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_nms[labels[i]])
        plt.axis("off")



#class_names[i]
print("Successfully terminated")
