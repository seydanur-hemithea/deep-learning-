# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:25:50 2025

@author: SEYDA NUR
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from keras.src.losses.loss import Loss


from keras.src.datasets import cifar10

from keras.src.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import RMSprop
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")



import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras


(x_train,y_train),(x_test,y_test)=cifar10.load_data()

class_labels=["Airplane","Automiobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]
fig,axes=plt.subplots(1,5,figsize=(15,10))
for i in range(5):
    axes[i].imshow(x_train[i])
    label=class_labels[int(y_train[i])]
    axes[i].set_title(label)
    axes[i].axis("off")
plt.show()

x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)


#%%
#data augmentation görüntüleri çeşşitlendirmek için kullanılır mevcut verilerden elde eldilir yakınlaştırp uzaklaştırmak gibi
datagen=ImageDataGenerator(rotation_range=20,
                           width_shift_range=0.2
                           ,height_shift_range=0.2
                           ,shear_range=0.2,zoom_range=0.2,
                           horizontal_flip=True,fill_mode="nearest")
datagen.fit(x_train)
#%%
#cnn modeli 
model=Sequential()
#conv layer,relu,convlayer,relu,pool,dropout,
#conv,rellu.conv,relu,pool,dropout
#classifcation=>flatten,dense,relu,dropout,dense(outputlayer)
#padding=filt görntü de dolaşıyoruz görüntünün boyunu azaltmamak iöin etrafını 0 larla donatıyoruz
model.add(Conv2D(32,(3,3),padding="same",activation="relu",
       input_shape=x_train.shape[1:]))
model.add(Conv2D(32,(3,3),activation="relu",))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))#bağlantıların %25 ini rastgele kapat

model.add(Conv2D(32,(3,3),padding="same",activation="relu"))
model.add(Conv2D(32,(3,3),activation="relu",))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))

model.summary()

model.compile(optimizer=RMSprop(learning_rate=0.0001,decay=1e-6)#LR =cnn ün öğrenme hızını b elirler küçük olursa yavaş öğrenir hızlı olursa öğrenemez decay değeri learnin rate i giderek düşürür
              ,loss="categorical_crossentropy",
              metrics=["accuracy"])

#model traing
history=model.fit(datagen.flow(x_train,y_train,batch_size=64),epochs=50,
          validation_data=(x_test,y_test))#doğrulama seti

#%%
#modelin test üzerinden tahmini yaptıralım
y_pred=model.predict(x_test)
y_pred_class=np.argmax(y_pred,axis=1)#arg max indexini alıcaz 
y_true=np.argmax(y_test,axis=1)

#classification report hesabı
report=classification_report(y_true,y_pred_class,target_names=class_labels)
print(report)

plt.figure()
plt.subplot(1,2,3)
plt.plot(history.history["Loss"],label="Train Loss",)
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training an validation Loss")
plt.lengend()
plt.grid()

plt.figure()
plt.subplot(1,2,3)
plt.plot(history.history["accuracy"],label="Train Accuracy",)
plt.plot(history.history["val_acc"],label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training an validation Accuracy")
plt.lengend()
plt.grid()
plt.tight_layout()
plt.show()







