# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:56:15 2025

@author: SEYDA NUR
"""
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist

from keras.src.layers import Dense
from keras.src.layers import Input

from keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim
from keras.models import Model

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
(x_train,_),(x_test,_)=fashion_mnist.load_data()
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0
plt.figure()
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(x_train[i],cmap="gray")
    plt.axis("off")
plt.show()
#vrilyi düzlestir.28*28 boyutundaki görüntüleri 784 boyutundaki vektöre cevir
x_train=x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test=x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
#%%autoencoder  icin model parametrelerinin tanimlanöasi
input_dim=x_train.shape[1]
encoding_dim=64
#encoder kisminin insaasi
input_image=Input(shape=(input_dim,))#girdir boyutunu  belirliyoruz
encoded=Dense(256,activation="relu")(input_image)#ilk gizli katman
encoden=Dense(128,activation="relu")(encoded)#ikinci gizli katman
encoded=Dense(encoding_dim,activation="relu")(encoded)#sıkıştırma katmani

decoded=Dense(128,activation="relu")(encoded)#☺genislentme katmaini
decoded=Dense(256,activation="relu")(decoded)#2. genislentme katmani
decoded=Dense(input_dim,activation="sigmoid")(decoded)#cikis katmani
#autoencoders olusturma encoder+decoder
Autoencoder=Model(input_image,decoded)#giristen cikisa tüm yapiyi tanimliyorux
#modelin compile edilmesi
Autoencoder.compile(optimizer=Adam(),loss="binary_crossentropy")

history=Autoencoder.fit(x_train,x_train,#girdi ve hedef ayni deger olmali buna otonom ögrenme denir
                        epochs=50,
                        batch_size=64,
                        shuffle=True,#egitim verileirni calistir
                        validation_data=(x_test,x_test),
                        verbose=1)#girdi ve hedef ayni deger olmali buna otonom ögrenme denir
#%%  mdoel test
#modeli encode rve decder olarak ikiye ayir
encoder=Model(input_image,encoded)
encoded_input=Input(shape=(encoding_dim,))
decoder_layer1=Autoencoder.layers[-3](encoded_input)
decoder_layer2=Autoencoder.layers[-2](decoder_layer1)
decoder_output=Autoencoder.layers[-1](decoder_layer2)

decoder=Model(encoded_input,decoder_output)
#testeverisi ile encoder  ve decoder ile sikıstirma ve yeniden yapilandirma 
encoded_images=encoder.predict(x_test)#latend temsili elde ederiz
decoded_images=decoder.predict(encoded_images)#latent temsillerini orginal forma geri döndür
#görsellestirme
n=10
plt.figure(figsize=(20,4))
for i in range(n):
    #orginal görüntü
    ax=plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28),cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax.plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_images[i].reshape(28,28),cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#ssim skorlarini hesapla
def compute_ssim(original,reconstructed):
    """
    her iki goruntu   arasinda ssim skoru 0 ile 1 arasinda 
    
    """
    original=original.reshape(28,28)
    reconstructed=reconstructed.reshape(28,28)
    return ssim(original,reconstructed,data_range=-1)
#veri seti icin ssim hesapla
ssim_score=[]
#iik 100 tanesini hesaplayalim
for i in range(100):
    original_img=x_test[i]
    reconstructed_img=decoded_images[i]
    score=compute_ssim(original_img,reconstructed_img)
    ssim_score.append(score)
    
average_ssim=np.mean(ssim_score)
print("SSİm:",average_ssim)

   

 
