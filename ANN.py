# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:48:09 2025

@author: SEYDA NUR
"""
#veri setinin hazırlanması
from keras.datasets import mnist#mnist load
from keras.utils  import to_categorical#categoric data trasfering
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential#sirali model
from keras.layers import Dense#bagli katmanlar

from keras.models import load_model

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

#mnist veri setinin yükle,egitim ve taest olarak ayır
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#ilk birkaç örneği görselleşitirelim
plt.figure(figsize=(10,5))
for i in range (6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i],cmap="gray")
    plt.title(f"index:{i},Label:{y_train[i]}")
    plt.axis("off")
plt.show()
 
#veri setini normalize edelim 0-255 pksel değerlerini 0 ile 1 arasına yerleştirellim
x_train=x_train.reshape((x_train.shape[0],
                         x_train.shape[1]*x_train.shape[2])).astype("float32")/255
x_test=x_test.reshape((x_test.shape[0],
                       x_test.shape[1]*x_test.shape[2])).astype("float32")/255

#etiketleri kategorik hale getir
y_train=to_categorical(y_train,10)#onehotencoding
y_test=to_categorical(y_test,10)

#ANN modelinin olusturlması ve derlenmesi
model=Sequential()
#ilk katman :512 tane cellden oluştur.relu aktivation function kullanacak ,input size:28,28=784
model.add(Dense(512,activation="relu",input_shape=(28*28,)))
# ikinci katman,256 cell ,activatin=tanh
model.add(Dense(256,activation="tanh"))

#relu: nonlineerite 1 den yukarıyuı kendi değerlerini atar 0 den küçükler i.in 0 a atar relu daha hzılı türevlenmesi kolay tanh daha yavaş bri ile 0 arasına yerleşitrir.
#ikiden fazla sınıf olduğu için çıkış katmanı activation=softmax olmalı ,ili sınıf olasaydı sigmoid fonksiuonu kullanacaktık.output layer 10 tane olamk zournda 10 sınıf oldupu için
model.add(Dense(10,activation="softmax")) 
model.summary()
#model derlemsei:optimizeer=adam adaptive momentum büyük veri ve komplex sinir ağları için idealdir.
#loss:categorical cross entropy sınıflandırma için yaygın olarak kullanılır
#metric"accuracy" :mokdelin başarıısnı doğruluk meriği ile ölçücez
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
#callbackler ve ann eğitilmesi
#earlystop:eğer val_los:validasyon veri setinden kaynaklanan kayıp değeri yani hata payı giderek azalmıyorsa öğrenme durmuş demektir.
early_stopping=EarlyStopping(monitor="val_loss",
                             patience=3,restore_best_weights=True)
#monitor=doğrulama setindeki kaybı izler,patience= 3 epoch boyunca val_loss değişmiyorsa erken durdurma yapılır,restore_best_weights=en iyi modelin ağırllıklarını geri yükler
#model checkpoint en iyi moddelin ağırlıklarını kaydededer
checkpoint=ModelCheckpoint("ann_best_model.keras",
                           monitor="val_loss",
                           save_best_only=True)
#traing=epoch=10,batchszie=60v,model 60000 veri swtine her biri 60 parçadan oluşan 10000 kere de train edecek buna epoch denir (10)eri setinin noral network e kaçar parçalar halinde verilecği,doğrulama seti oranı=%20
#valsplit için ayrılan %20 lik kısım ayrıldığında 800 kerede train edecek
history=model.fit(x_train,y_train,
          epochs=10,batch_size=60,
          validation_split=0.2,#eigitim verisin %20 sini doğrulama için ayrıacak
          callbacks=[early_stopping,checkpoint])
# model evalution,visiulation,model save and load
#test verisi ile model performansı değerlendirme
#evalute:modelin test verisi üzerindeki loss (test_loss) ve accuracy (test_CC) HESAPLAR
test_loss,test_acc=model.evaluate(x_test,y_test)
print(f"Test acc:{test_acc},test loss:{test_loss}")
#trainin ve validasyon accuracy görselleşitr
plt.figure()
plt.plot(history.history["accuracy"],marker="o",label="Trainin Accuracy")
plt.plot(history.history["val_accuracy"],marker="o",label="Validation Accuracy")
plt.title("ANN Accuracy on MNİST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.grid(True)
plt.show()
#trainin and validation loss görselleştirme
plt.figure()
plt.plot(history.history["loss"],marker="o",label="Trainin Loss")
plt.plot(history.history["val_loss"],marker="o",label="Validation Loss")
plt.title("ANN Loss on MNİST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

#model kaydet
model.save("final_mnist_ann_model.keras")

loaded_model=load_model("final_mnist_ann_model.keras")

loaded_model=load_model("final_mnist_ann_model.keras")
test_loss,test_acc=loaded_model.evaluate(x_test,y_test)
print(f"loaded model rseult-> Test acc:{test_acc},test loss:{test_loss}")
