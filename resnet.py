# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:16:21 2025

@author: SEYDA NUR

"""

import tensorflow as tf


from keras.layers import Conv2D,BatchNormalization,Activation,Add,Flatten,Dense,Input
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
from kerastuner import HyperModel,RandomSearch#hyperparmaeter tuner icin
#veri setini yukle
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
#veri setini reshape
train_images=train_images.reshape(-1,28,28,1).astype("float32")/255.0
test_images=test_images.reshape(-1,28,28,1).astype("float32")/255.0

train_labels=to_categorical(train_labels,10)
test_labels=to_categorical(test_labels,10)
#%%residual bloc
def residual_block(x,filters,kernel_size=3,stride=1):
    shortcut=x
    #1.conv katmanı
    x=Conv2D(filters,kernel_size=kernel_size,strides=stride,padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    
    #2.conv katmani
    x=Conv2D(filters,kernel_size=kernel_size,strides=stride,padding="same")(x)
    x=BatchNormalization()(x)
    #eger gristen gelen verilerim  boyutu filtre sayisina esit degilse
    if shortcut.shape[-1]!=filters:
        #gris verisinin boyutunu esitlemek icin 1x1 konvolusyon uygulayalım
        shortcut=Conv2D(filters,kernel_size=1,strides=stride,padding="same")(shortcut)
        shortcut=BatchNormalization()(shortcut)
    #redisual bagalnti: giris verisi ile cikis verisini toplayalim
    x=Add()([x,shortcut])
    x=Activation("relu")(x)
    
    return x
#%%
class ResNetModel(HyperModel):
    def build(self,hp):#hp=hyper pRMETER TINİNG İCİN KULLANİLACAK
        inputs=Input(shape=(28,28,1))
        #1. cov layer
        x=Conv2D(filters=hp.Int("initial_filters",min_value=32,max_value=128,step=32),
                 kernel_size=3,padding="same",activation="relu")(inputs)
        x=BatchNormalization()(x)
        
        #residual bloc ekleyelim
        for i in range(hp.Int("num_blocks",min_value=1,max_value=3,step=1)):
            x=residual_block(x,hp.Int("res_filters"+str(i),min_value=32,max_value=128,step=32))
            
        #siniflandirma katmanş
        x=Flatten()(x)
        x=Dense(128,activation="relu")(x)
        outputs=Dense(10,activation="softmax")(x)
        model=Model(inputs,outputs)
        model.compile(optimizer=Adam(hp.Float("learning_rate",min_value=1e-4,max_value=1e-2,sampling="LOG")),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model
#%%
tuner=RandomSearch(
    ResNetModel(),
    objective="val_accuracy",#tuning referans degeri
    max_trials=2,#en az 100 kere denenmsei gerkiyor
    executions_per_trial=1,#her denemde kac kere egitim yapilacagi)
    directory="resnet_hyperparameter_tuning_directory",
    project_name="resnet_model_tuning.h"
    )
 #hyperrparameter optimizasyonu and training
tuner.search(train_images,train_labels,epochs=1,validation_data=(test_images,test_labels))       
best_model=tuner.get_best_models(num_models=1)[0]
#en iyi modeli test edliö
test_loss,test_acc=best_model.evaluate(test_images,test_labels)
print(f"Test loss:{test_loss:.4f},test accuracy:{test_acc:.4f}")        
        
