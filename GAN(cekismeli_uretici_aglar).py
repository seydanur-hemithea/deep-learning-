# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:01:30 2025

@author: SEYDA NUR
"""
#Generative Adversarial Network
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.datasets import mnist#mnist load
from keras.models import Sequential#sirali model
from keras.layers import LeakyReLU#olu noronlar sorunu :negatif girslerde sıfıra sabitlenirse ogrenme olumsuz etkilenebilir bu yuzden leakyrelu tanimlanmistir.

from keras.layers import Dense,Reshape,Flatten,Conv2D,Conv2DTranspose,BatchNormalization
from keras.optimizers import Adam
from tqdm import tqdm#for dongusune ilerleme sayaci ekler

import warnings
warnings.filterwarnings("ignore")

(x_train,_),(_,_)=mnist.load_data()
x_train=x_train/255.0
x_train=np.expand_dims(x_train,axis=-1)
#%%
#discriminator and generator create
#gan parametreleri
z_dim=100 #gurultu vektorunun boyutu
def build_discriminator():
    model=Sequential()
    #Conv2D:64 filtre,3x3 cekirdek kernel,stride=2(filtreyi kac piksel atlayarak kullanacagi),padding=same,activation=leakyrelu,
    model.add(Conv2D(64,kernel_size=3,strides=2,padding="same",input_shape=(28,28,1)))
    model.add(LeakyReLU(alpha=0.2))
    #Conv2D:64 filtre,3x3 cekirdek kernel,stride=2,padding=same,activation=leakyrelu,
    model.add(Conv2D(128,kernel_size=3,strides=2,padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    #flatten=output layer,
    model.add(Flatten())#goruntuyu tek boyutlu vektore donusturur
    model.add(Dense(1,activation="sigmoid"))
    #compile
    model.compile(loss="binary_crossentropy",optimizer=Adam(0.0002,0.5),metrics=["accuracy"])   
    return model
         
def build_generator():
    model=Sequential()
    model.add(Dense(7*7*128,input_dim=z_dim))#gurultu vektprlerinde yuksek boyutlu uzaya donusum
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7,7,128)))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64,kernel_size=3,strides=2,padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(1,kernel_size=3,strides=2,padding="same",activation="tanh"))
    return model

  #%%
def build_gan(generator,discriminator):
    discriminator.trainable=False#discriminator egitilemez
    model=Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss="binary_crossentropy",optimizer=Adam(0.0002,0.5))
    return model


generator=build_generator()
discriminator=build_discriminator()
gan=build_gan(generator,discriminator)
print(gan.summary())
#%%train



epochs=100000
batch_size=64
half_batch=batch_size//2

for epoch in tqdm(range(epochs),desc="Training Process"):
   
    #gercek veriler ile discriminator egitim, 
    idx=np.random.randint(0,x_train.shape[0],half_batch)
    real_images=x_train[idx]
    real_label=np.ones((half_batch,1))                  
    #fake veriler (generator uretecek) ile discriminator egitimi yapilacak
    noise=np.random.normal(0,1,(batch_size,z_dim))#gurultu  vektorleri
    fake_images=generator.predict(noise,verbose=0)
    fake_label=np.zeros((half_batch,1))
 
    #update discrimintor
    #trainon bach mini batch veri ile eigitim tek seferlik,gercek veriler ile kayıp heasplama
 
    d_loss_real=discriminator.train_on_batch(real_images,real_label)
    d_loss_fake=discriminator.train_on_batch(fake_images,fake_label)#sahte verilerle kayip hesaplama
    d_loss=np.add(d_loss_real,d_loss_fake)*0.5
    
    noise=np.random.normal(0,1,(batch_size,z_dim))
    valid_y=np.ones((batch_size,1))#dogru etiketler
    g_loss=gan.train_on_batch(noise,valid_y)
   #train gam
    if epoch %100 ==0:
        print(f"{epoch}/{epoch} D Loss:{d_loss[0]},G Loss:{g_loss}")
       
   #gan ın içinde bulunan generatorun egitimi
    #%%
def plot_generated_images(generator,epoch,examples=10,dim=(1,10)):
    noise=np.random.normal(0,1,(examples,z_dim))
    gen_images=generator.predict(noise,verbose=0)
    gen_images=0.5*gen_images+0.5
    
    plt.figure(fig_size=(10,1))
    for i in range(gen_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        plt.imshow(gen_images[i,:,:,0],cmap="gray")
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()
plot_generated_images(generator,epochs)

                 
                 
