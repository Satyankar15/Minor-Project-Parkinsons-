# -*- coding: utf-8 -*-
"""
Created on Mon May 24 00:06:34 2021

@author: satya
"""
import cv2
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
import random
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

os.chdir(r"D:/Minor Project 2021/Image Classifier")
pick_in=open('data1.pickle','rb')
data=pickle.load(pick_in)
pick_in.close()     
print('Length: ',len(data))


features=[]
labels=[]
random.shuffle(data)


for feature,label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features,labels, test_size=0.33)
 
#print(features[0])
#print(labels[0])
param_grid={'C':[0.05,0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)
#maxacc=0
#c=0
#maxgamma=0
#for z in [0.05,0.1,1,10,100,1000,10000]:
    #for gamma in np.arange(0.001, 0.101, 0.001):
#model=SVC(kernel='poly',C=z, gamma='auto')
model.fit(xtrain,ytrain)
pred=model.predict(xtest)

accuracy=accuracy_score(ytest,pred)
#if(accuracy>maxacc):
 #       maxacc=accuracy
  #      c=z
        #maxgamma=gamma

print('Accuracy: ',accuracy)
#print('C: ',c) 
#print('Gamma: ',maxgamma)  