# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:27:20 2025

@author:SEYDA NUR

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import Model
from keras.layers import RandomZoom, RandomFlip, RandomRotation
from keras.applications import MobileNetV2
from pathlib import Path
import os.path
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.layers import Resizing, RandomFlip, RandomRotation, Rescaling
from keras.models import Sequential
dataset = "Drug Vision/Data Combined"
image_dir = Path(dataset)
filepaths = list(image_dir.glob(r"**/*.jpg"))+list(image_dir.glob(r"**/*.png"))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
filepaths = pd.Series(filepaths, name="filepath").astype("str")
labels = pd.Series(labels, name="label").astype("str")
image_df = pd.concat([filepaths, labels], axis=1)
random_index = np.random.randint(0, len(image_df), 25)
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(11, 11))
for i, aks in enumerate(axes.flat):
    aks.imshow(plt.imread(image_df.filepath[random_index[i]]))
    aks.set_title(image_df.label[random_index[i]])
plt.tight_layout()
# %%

train_df, test_df = train_test_split(
    image_df, test_size=0.2, random_state=42, shuffle=True)
# ☺data augmentation
 # veri artırımı
train_generator = ImageDataGenerator(
    # mobile net icin ön islemee
    preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
                                   validation_split=0.2)
test_generator = ImageDataGenerator(
    # mobile net icin ön islemee
    preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
                                   )
# egitim verilerin, data frameden akısa alalım
train_images = train_generator.flow_from_dataframe(dataframe=image_df,
                                                 x_col="filepath",
                                                 y_col="label",
                                                 target_size=(224, 224),
                                                 color_mode="rgb",
                                                 class_mode="categorical",
                                                 batch_size=64,
                                                 shuffle=True,
                                                 seed=42,
                                                 subset="training")
val_images = train_generator.flow_from_dataframe(dataframe=train_df,
                                               x_col="filepath",
                                               y_col="label",
                                               target_size=(224, 224),
                                               color_mode="rgb",
                                               class_mode="categorical",
                                               batch_size=64,
                                               shuffle=True,
                                               seed=42,
                                               subset="validation")
test_images = test_generator.flow_from_dataframe(dataframe=train_df,
                                               x_col="filepath",
                                               y_col="label",
                                               target_size=(224, 224),
                                               color_mode="rgb",
                                               class_mode="categorical",
                                               batch_size=64)
resize_and_rescale = tf.keras.Sequential([keras.preprocessing.Resizing(224, 224),
                                        keras.preprocessing.Rescaling(1.0/255)])

# %%mobile net önceden egitilmiş model
pretrained_model = MobileNetV2(input_shape=(224, 224, 3),  # girdilerin boyutu
                             include_top=False,  # mobile net sınıflandırma katmanını dahil etme
                             weights="imagenet",  # hangi veri setiyle train edildigi
                             pooling="avg")
pretrained_model.trainable = False
checkpoint_path = "pharmaceutical_drugs_and_vitamins_classification_model_checkpoint.weights.h5"
checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True,
                                    monitor="val_accuracy",
                                    save_best_only=True,
                                    )
early_stopping = EarlyStopping(monitor="val_accuracy",
                             patience=5,
                             restore_best_weights=True)
inputs = pretrained_model.input
x = resize_and_rescale(inputs)
x = Dense(256, activation="relu")(pretrained_model.output)
x = Dropout(0.2)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(10, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adam(0.0001),
              loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_images, steps_per_epoch=len(train_images), validation_data=val_images,
                  validation_steps=len(val_images),
                  epochs=10,
                  callbacks=[early_stopping, checkpoint_callback])
# %%test
loss, accuracy = model.evaluate(test_images, verbose=1)
print(f"loss:{loss:.4f},accuracy:{accuracy:.4f}")

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], marker="o", label="Trainin Accuracy")
plt.plot(history.history["val_accuracy"],
         marker="o", label="Validation Accuracy")
plt.title("trainin and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.grid(True)
plt.show()
# trainin and validation loss görselleştirme

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], marker="o", label="Trainin Loss")
plt.plot(history.history["val_loss"], marker="o", label="Validation Loss")
plt.title("trainin and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


prd = model.predict(test_images)
prd = np.argmax(prd, axis=1)
labels = (train_images.class_indices)
labels = dict((v, k)for k, v in labels.items())
prd = [labels[k]for k in prd]

random_index = np.random.randint(0, len(test_df)-1, 15)
fig, axes = plt.subpplots(nrows=5, ncols=3, figsize=(11, 11))

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.filepath[random_index[i]]))
    if test_df.label.iloc[random_index[i]] == prd[random_index[i]]:
        color = "green"
    else: color = "red"

    ax.set_title(f"True:{test_df.label.iloc[random_index[i]]}\n predicted:{prd[random_index[i]]}")
    
plt.tight_layout()

y_test=list(test_df.label)
print(classification_report(y_test,prd))
