
import cv2
import os
import numpy as np
import pickle
            
dir =r"D:/Minor Project 2021/Image Classifier/Images"


data=[]
        
categories=['NP','PP']


for category in categories:
    path=os.path.join(dir,category)
    print(path)
    label=categories.index(category)
    print(label)
    for img in os.listdir(path):
        imgpath=os.path.join(path,img)
        pet_img=cv2.imread(imgpath,0)
        pet_img=cv2.resize(pet_img,(150,150))
        image=np.array(pet_img).flatten()

        
        data.append([image,label])
        
print(len(data))
os.chdir(r"D:/Minor Project 2021/Image Classifier")
pick_in=open('data1.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()