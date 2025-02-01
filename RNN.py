# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:54:04 2025

@author: SEYDA NUR
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from keras.src.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,SimpleRNN,Dense,Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report,roc_curve,auc

import kerastuner as kt
from kerastuner.tuners import RandomSearch


import warnings
warnings.filterwarnings('ignore')

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=10000)

#%%veri önişleme
#padding  yorumları aynı uzunluğa getirmek için yappılır
maxlen=100
x_train=pad_sequences(x_train,maxlen=maxlen)

x_test=pad_sequences(x_test,maxlen=maxlen)
#%%RNN modelini bir fonsiyon içinde tanımlayalım
def build_model(hp):  #hiperprameter
   model=Sequential()#base model
   #embeding katmanı kelimeleri vektörlerre dönüştürür,simplearnn,dropout katmanı,çıkış katmanı 1 cell ve sigmoid
   model.add(Embedding(input_dim=10000,output_dim=hp.Int("embedding_output",
                                                         min_value=32,max_value=128,
                                                         step=32),input_length=maxlen))#modelimiz trainin edilirken emb.output 32 128 arasında değişecek ,değer olarak vektör boyutları 32,64,96,128 olabilir.
   model.add(SimpleRNN(units=hp.Int("rnn units",min_value=32,max_value=128,step=32)))#rnn hücre sayysı32,64,96,128 olabilr
   
   model.add(Dropout(rate=hp.Float("dropout_rate",min_value=0.2,max_value=0.5,step=0.1)))
   
   model.add(Dense(1,activation="sigmoid"))#binary classification yani 2li sınıflandırma için sigmoid kullanılır
   
   model.compile(optimizer=hp.Choice("optimizer",["adam","rmsprop"]),
                 loss="binary_crossentropy",#ikili sınıflandırma için kullanılan loss fonksiyonu
                 metrics=["accuracy","AUC"])#auc=area under curve
   
   return model
#%%hiperparameter search:random search
tuner=RandomSearch(build_model,#optimize edilecek model fonksiyonu
                   objective="val_loss",#val loss en dusuk olan en iyisidirr
                   max_trials=2,
                   executions_per_trial=1,#her model için bir eğitim denemesi yapılacak
                   directory="rnn_tuner_directory",#modellerin kayit edilecegi dizin
                   project_name="imdb_rnn"
                   )
early_stopping=EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)
tuner.search(x_train,y_train,
             epochs=2,
             validation_split=0.2,
             callbacks=[early_stopping])

#%%evaluate best model
#en iyi modelin alinmasi
best_model=tuner.get_best_models(num_models=1)[0]
loss,accuracy,AUC_score=best_model.evaluate(x_test,y_test)

print(f"test loss:{loss},test accuracy:{accuracy:.3f},test auc:{AUC_score:.3f}")

y_pred_prob=best_model.predict(x_test)
y_pred=(y_pred_prob>0.5).astype("int32")
print(classification_report(y_test,y_pred))
#roc egrisi hesaplama
fpr,tpr,_=roc_curve(y_test,y_pred_prob)#false positive rate,false postive rate
roc_auc=auc(fpr,tpr)
#roc egriisin altında kalan alan hesaplanır
#roc egrisi görselleştir
plt.figure()
plt.plot(fpr,tpr,color="darkorange",lw=2,label="ROC curve(area=%0.2f)"% roc_auc)
plt.plot([0,1],[0,1],color="blue",lw=2,linestyle="--")
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.xlabel("FpR")
plt.ylabel("TpR")
plt.title("Receiver operating characteristic ROC curve")
plt.lengend()
plt.show()
