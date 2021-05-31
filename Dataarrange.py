# -*- coding: utf-8 -*-
"""
Created on Sat May 15 23:04:59 2021

@author: satya
"""
import cv2
import os
import numpy as np
import pickle
            
dir =r"D:/Minor Project 2021/Image Classifier/Images"

#print(os.listdir(dir))
#with disable_file_system_redirection():
    #print(os.listdir(dir))
#print(os.listdir(dir))

#print(os.getcwd())
data=[]
        
categories=['NP','PP']

#print(os.walk(dir))
for category in categories:
    path=os.path.join(dir,category)
    print(path)
    label=categories.index(category)
    print(label)
    for img in os.listdir(path):
        imgpath=os.path.join(path,img)
        pet_img=cv2.imread(imgpath,0)
        #pet_img = cv2.cvtColor(pet_img, cv2.COLOR_BGR2GRAY)
        pet_img=cv2.resize(pet_img,(150,150))
        image=np.array(pet_img).flatten()

        
        data.append([image,label])
        
print(len(data))
os.chdir(r"D:/Minor Project 2021/Image Classifier")
pick_in=open('data1.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()