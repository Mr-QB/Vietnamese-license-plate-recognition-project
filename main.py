import cv2
import library
import numpy as np

################## PLATE #################
img = cv2.imread('D:/test/test.png')#read pictures 
contour = library.pretreatment(img) # extract contour of the image
list_img_plate,cnt_plate = library.detect_plate(contour,img) #detect images that may be plate 

####################### CHAR ##########################
img_filter_plate,contour_char,list_img_char,stt_cnt = library.detect_char(list_img_plate)
imgOrigin = img_filter_plate.copy()
for i in contour_char:
    imgOrigin = library.drawBoundingBox(imgOrigin, i)
print(contour_char)
#cv2.imshow('end',imgOrigin)
#cv2.waitKey()
imgOrigin = library.drawBoundingBox(img, cnt_plate[stt_cnt])
#imgOrigin=np.uint8(imgOrigin)
char = library.train_char(list_img_char)
library.write_text(img,cnt_plate[stt_cnt],char)
cv2.imshow('end',imgOrigin)
cv2.waitKey()
'''for img in list_img_char:
    cv2.imshow('img1',img)
    cv2.waitKey()'''
print("License plate has the form : ",char)
