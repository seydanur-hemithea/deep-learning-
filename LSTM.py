# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 17:54:28 2025

@author: SEYDA NUR
"""
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

import keras
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout
from keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

newsgroup= fetch_20newsgroups(subset='all') #hem eğitim hem test verilerini yüklüyoryz

X=newsgroup.data#x=metinler
y=newsgroup.target#y=metinlere karsilik gelen etiketler
#metin verilerini tokenize etme ve padding
tokenizer=Tokenizer(num_words=10000)#en cok kullanilan kelime sayisi
tokenizer.fit_on_texts(X)

X_sequences=tokenizer.texts_to_sequences(X)#metinleri sayisala cevir
X_padded=pad_sequences(X_sequences,maxlen=100)
                       #etiketleri encoding yapalim sayisal hale dönüstürelim
label_encoder=LabelEncoder()
y_encoded=label_encoder.fit_transform(y)                    
                       
X_train,X_test,y_train,y_test=train_test_split(X_padded,y_encoded,test_size=0.2,random_state=42)
#%%
from keras import beckend as K
def f1_score(y_true,y_pred):#metric
    y_pred=K.round(y_pred)
    tp=K.sum(K.cast(y_true*y_pred,"float"),axis=0)
    fp=K.sum(K.cast((1-y_true)*y_pred,"float"),axis=0)
    fn=K.sum(K.cast(y_true*(1-y_pred),"float"),axis=0)
    
    precision=tp/(tp+fp+K.epsilon())
    recall=tp/(tp+fn+K.epsilon())
    f1=2*(precision*recall)/(precision+recall+K.epsilon)
    
    return K.mean(f1)


#create builld LSTM
def build_lstm_model():
    
    model=Sequential()
    
    model.add(Embedding(input_dim=10000,output_dim=64,input_length=100))
    #inpu_dim=kelime vektorlerinin toplam boyutu,outpu_dim=kelime vektorlerinin boyut,input_length=her bir gitis metninin uzunllugu
    #lstm katmani
    model.add(LSTM(units=64,return_sequences=False))
    #64 adet hücre,
    #return_seqeunce=sonuclarin tüm zaman adimlari yerine sadece son adimda return etmesi normalde her lstm hücresinin kendi ciktisi vardi,
    #dropout katmani
    model.add(Dropout(0.5))
    #dense karmani(cikis katmai)
    model.add(Dense(20,activation="softmax"))
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy"#cok sinifli siniflandirma problemleri icin kayip fonksi
                  ,metrics=["accuracy",f1_score]
                  )
    return model
#model olusturma
model=build_lstm_model()
model.summary()
#%%
early_stopping=EarlyStopping(monitor="val_accuracy",patience=5,
                            restore_best_weights=True)
history=model.fit(X_train,y_train,
                  epochs=5,batch_size=32,
                  validation_split=0.1,callbacks=[early_stopping])


#%% model evaluation
loss,accuracy=model.evaluate(X_test,y_test)
print(f"Test Loss:{loss:.2f},Test Accuracy:{accuracy:.2f}")

plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.grid("True")


plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Loss")
plt.legend()

plt.grid("True")

plt.show()
                      
                       
                       
                       
                       
                       
                       
                       
                       
                       