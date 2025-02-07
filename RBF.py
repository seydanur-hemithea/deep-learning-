# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:05:44 2025

@author: SEYDA NUR
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.src.layers import Layer,Dense,Flatten
from keras import backend as K
import keras 
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

iris=load_iris()
X=iris.data
y=iris.target

label_binarizer=LabelBinarizer()#one hot endcoding formatına donustureen nesne
y_encoded=label_binarizer.fit_transform(y)
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)


X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_encoded,test_size=0.2,random_state=42)
#%%
#RBF katmanı
class RBFLayer(Layer):#rbf layer kerasın layer sınıfından miras alır 
    def __init__(self,units,gamma,**kwargs ):
        """"
        contructer,
        katmanin genel ozelliklerini baslatmak icin gereklidir
        
        """
        super(RBFLayer,self).__init__(**kwargs)#layer sinifinin init metodunun cagiir,katmanin genel ozelliklerini baslatmak icin gereklidir
        self.units=units#rbf katmaninda gizli noron sayisi
        self.gamma=K.set_floatx('float32')#rbf fonksiyonu yayilşim parametresi rbf duyarliligi,gammayi keras float32 turune donusturr
        
    def build(self,input_shape):
        """
        build metodu katmanin agirliklarini tanimlar
        bu metot keras tarafindan katman ilk defa bir input aldiginda otomatik olarak cagrilir
        
        """
        #add_weight kerasta egitilebilecek agirliklari tanimlamak icin kullanilir
        self.mu=self.add_weight(name="mu",
                                shape=(int(input_shape[1]),self.units),#shape=agirliklarin boyutunu tanimlar,input_shape[1]=girs verisinin boyutu,self.units=merkezlerin sayisi,
                                initializer="uniform",#agirliklarin baslangic degeri belirlenir
                                trainable=True#agirliklar egitilebilir.
                                )
        super(RBFLayer,self).build(input_shape)#layer sinifinin build metodunu cagrlir katmanin insaasi tamamlanir
    def call(self,inputs):
        """
        katman cagrildiginda (yeni forward propogation sirasinda)calisir
        bu fonksşyon girdiyi alır ,ciktiyi hesaplar
        
        
        """
        diff=K.expand_dims(inputs) - self.mu
         #K.expend_dims(inputs girdiye bir boyut ekler
        L2=K.sum(K.pow(diff,2),axis=1)#K.pow(diff,2)=diffin karesi,K.sum()=farklarin toplami alinir ve L2 normu hesaplanir
        res=K.exp(-1*self.gamma*L2)#K.experbf=exp(-gamma*L2)
        #L2 mesafesinin gamma ile carpilmasi ve negatigf bir  usstel fonksiyonun alinmasi rbf degerini uretir
        return res
       
    def compute_output_shape(self,input_shape):
        """
    
        bu metot katmanin citisinin sekli hakkinda bilgi verir
        keras yardimci fonksiyonlarindan bir tanesidr
        """
        return (input_shape[0],self.units)#ciktinin sekli (num_samples,num_units).input_shape[0]=sample sayisi
#%%
def build_model():
    model=Sequential()
    model.add(Flatten(input_shape=(4,)))#griş verisini düzleştir
    model.add(RBFLayer(10,0.5))#RBFkatmani(10 noron gamma0.5
    model.add(Dense(3,activation="softmax"))#output katmani 3 sinif var
    
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
#model olusturalım
model=build_model()
history=model.fit(X_train,y_train,
                  epochs=250,
                  batch_size=4,
                  validation_split=0.3,
                  verbose=1)
#%%
loss,accuracy=model.evaluate(X_test,y_test)

plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["Loss"],label="Train Loss",)
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training an validation Loss")
plt.lengend()
plt.grid(True)
    
plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],label="Train Accuracy",)
plt.plot(history.history["val_acc"],label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training an validation Accuracy")
plt.lengend()
plt.grid(True)

plt.show()
    
    

      
  
    