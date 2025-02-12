# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:53:35 2025

@author: SEYDA NUR
"""
import gym#rain forcement learning icin env saglar ,gelistrme ortami saglar
import numpy as np
from collections import deque#ajanin bellegini tanimlamak icin gerekli deque veri yapisi
from keras.models import Sequential#sirali model olusturmak icin

from keras.layers import Dense#tam bagalantili nn
from keras.optimizers import Adam
import random


#for döngüsünü progress bar a dönüstürebilmek ilerlemeyi görsellestirmek icin
from tqdm import tqdm
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
#dqla class
class DQLAgent:
    def __init__(self,env):#parametrelşeri ve hper parametreleri tanimla env :cartpole
        #cevrenin gözlem alani(state)  boyut sayisi
        self.state_size=env.observation_space.shape[0]
        #cevrede bulunan eylem sayisi(ajanin secebileceği eylem sayisi)
        self.action_size=env.action_space.n
        #geleckteki ödğllerin indirm orani
        self.gamma=0.95
        #learning rate ajanin ögrenme hizi
        self.learning_rate=0.001
        #kesfetme orani=epsilon=1 olsun öax kesif cevreyi once kefsfetmek zzournda oyuzden yüksek
        
        self.epsilon=1
        #her iterasyonda epsilon ne kadr azalacak,epsilon azaldıkça daha fazla ögrenme daha az kesif
        self.epsilon_decay=0.995
        #min kesfetme orani
        self.epsilon_min=0.01
        #AJANİN DENYİmleri
        self.memory=deque(maxlen=1000)
        #dein ogrenem modelini insaa et 
        self.model=self.build_model()
    def build_model(self):#deep q learning sinir aglarini olustur
        model=Sequential()
        #girdi katmani 48 noron relu activ
        model.add(Dense(48,input_dim=self.state_size,activation="relu"))
        model.add(Dense(24,activation="relu"))
        #output katmani
        model.add(Dense(self.action_size,activation="linear"))#kesin degerler dönmesi gerek
        model.compile(loss="mse",optimizer=Adam(self.learning_rate))#lineer kullandıgimiz iicn mse 
        return model
    def remember(self,state,action,reward,next_state,done):#ajanin deneyimlerini bellek veri yapisina kaydet
        self.memory.append((state,action,reward,next_state,done))
    def act(self,state):#ajanimiz eylem secebilecek
        #eger rastgelen uretilen sayi epsilondan kucukse rastegele eyelem secilir kesif yapilir
        if random.uniform(0,1)<=self.epsilon:
            return env.action_space.sample()#rastgele eylem sec 
        #aksi durmda model tarafinda tahmin edilen degerlere göre eneiy eylem secilri
        act_values=self.model.predict(state,verbose=0)
        return np.argmax(act_values[0])
    def replay(self,batch_size):#deneyeimleri tekarar oynatarak deep q agi egitilir
        
        #bellekte yeterince denyim yoksa geri oynatma yapilmaz 
        if len(self.memory)<batch_size:
            return 
      
            #bellekten rastgele batch szie kadar deneyim sec
        minibatch=random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in minibatch:
            if done:#eger done ise bitis durumunda odulu dogrudan hedef alır 
                target=reward
            else:
                target=reward+self.gamma*np.amax(self.model.predict(next_state,verbose=0)[0])
            #modelin tahmin ettiği oduller
            train_target=self.model.predict(state,verbose=0)
            #ajanin yaptigi eyleme göre tahmin edilen odulu guncelle
            train_target[0][action]=target
            #modeli egit
            self.model.fit(state,train_target,verbose=0)
            
    def adaptiveEGreedy(self):#epsilonun zamanla azalmasi yani kesif,somuru dengesi
    
        if self.epsilon> self.epsilon_min:
            self.epsilon=self.epsilon*self.epsilon_decay
#%%
env=gym.make("CartPole-v1",render_mode="human")#cartpole ortamini baslatiyoryz
agent=DQLAgent(env)
batch_size=32#egitim icin minibatch byutu
episodes=5#epoch,similasyonun oynatılacagi toplam bolum sayisi
for e in tqdm (range(episodes)):
    #ortami sifirla ve baslangic durumunu al 
    state=env.reset()[0]#ortami sifirlamak
    state=np.reshape(state,[1,4])
    
    time=0#zman adimini baslat
    
    while True:
        #ajan eylem secer
        action=agent.act(state)
        #ajanşmiz ortamda bu eylemi uuygular ve bu eylem sonucunda next_state ,reward,bitis bilgisi alir
        (next_state,reward,done,_,_)=env.step(action)
        next_state=np.reshape(state,[1,4])
        #yapmis oldugu bu adimi ayni eylemi ve bu eylem sonucu envdan alinan bilgileri kayder
        agent.remember(state,action,reward,next_state,done)
        #mevcut durumu günceller
        state=next_state
        #deneyimlerden yeniden oynatmayi baslatirrepley=trainig
        agent.replay(batch_size)
        #epsilonu set eder
        agent.adaptiveEGreedy()
        #zaman adimini artırır
        time=time+1
        #done ise donguyu kirar ,bolum biter ve yeni bolume baslar
        if done:
            print(f"Episode:{e},time:{time}")
            break
#%%
import time

trained_model=agent#egitm model, al

env=gym.make("CartPole-v1",render_mode="human")
state=env.reset()[0]
state=np.reshape(state,[1,4])
time_t=0
while True:
    env.render()#ortami gmrsel oalrak render et
    action=trained_model.act(state)#egitilen modelden action gerceklerşitr
    (next_state,reward,done,_,_)=env.step(action)#eylem uygula
    next_state=np.reshape(state,[1,4])
    time_t +=1
    print(f"time:{time_t}")
    time.sleep(0.5)
    
    if done:
        break
print("Done")  
    



            
            
        