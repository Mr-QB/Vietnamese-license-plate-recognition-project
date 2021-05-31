import cv2
import numpy as np
import os

from numpy.core.fromnumeric import size
folder_data_1 = 'D:/data/data_1'
folder_data_2 = 'D:/data/data_2'

w,h=20,20    ##### size ######

def load_images_from_folder(folder):
    images = []
    sum = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        img = cv2.resize(img, (w,h))  
        img = np.array(img).astype(np.float32)
        img = np.reshape(img,-1)
        if img is not None:
            images.append(img)
            sum+=1
    return images,sum

data_train=[]
def update_data(folder_1,folder_2):
    data_1,data_2 = [],[]
    data_labels = [0]
    for folder in os.listdir(folder_1):
        folder_1a = folder_1 + '/' + folder
        images,sum = load_images_from_folder(folder_1a)
        data_labels = np.concatenate((data_labels,np.repeat(ord(folder),sum)))
        data_1.append(images)                  
    ################################     
    for folder in os.listdir(folder_2):
        folder_2a = folder_2 + '/' + folder
        images,sum = load_images_from_folder(folder_2a)
        data_labels = np.concatenate((data_labels,np.repeat(ord(folder),sum)))
        data_2.append(images)
    ################################
    data_a = np.concatenate((data_1,data_2))
    data = data_a[0]
    for i in range(1,len(data_a)):
        data = np.concatenate((data,data_a[i]))
    return data,data_labels.astype(np.float32)[1:]
    
    

data_train,data_labels = update_data(folder_data_1,folder_data_2)
#print(len(data_train))
#print(len(data_labels))

#update test:
img = cv2.imread('D:/test.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgROIResized = cv2.resize(img_gray, (w,h))
npaROIResized = imgROIResized.reshape((1, w*h))
npaROIResized = np.float32(npaROIResized)

#train:
kNearest = cv2.ml.KNearest_create()
kNearest.setDefaultK(1)                    
np.savetxt('data_train.txt', data_train)
np.savetxt('data_train_labels.txt', data_labels)                                   
kNearest.train(data_train, cv2.ml.ROW_SAMPLE, data_labels) 
        #kq1,kq,kq2,kq3 = kNearest.findNearest(npaROIResized,1)
retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,k=1)  
strCurrentChar = str(chr(int(npaResults[0][0])))  
print('thanhcong')
