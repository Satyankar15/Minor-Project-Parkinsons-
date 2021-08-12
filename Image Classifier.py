
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
 

param_grid={'C':[0.01,0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['poly'],'degree':[4,5,6,7]}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)

model.fit(xtrain,ytrain)
pred=model.predict(xtest)

accuracy=accuracy_score(ytest,pred)


print('Accuracy: ',accuracy)
