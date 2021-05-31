import cv2
import thuvien as tv 
import numpy as np

################## PLATE #################
img = cv2.imread('D:/test/test.png')#read pictures 
contour = tv.pretreatment(img) # extract contour of the image
list_img_plate,cnt_plate = tv.detect_plate(contour,img) #detect images that may be plate 

####################### CHAR ##########################
img_filter_plate,contour_char,list_img_char,stt_cnt = tv.detect_char(list_img_plate)
imgOrigin = img_filter_plate.copy()
for i in contour_char:
    imgOrigin = tv.drawBoundingBox(imgOrigin, i)
print(contour_char)
#cv2.imshow('end',imgOrigin)
#cv2.waitKey()
imgOrigin = tv.drawBoundingBox(img, cnt_plate[stt_cnt])
#imgOrigin=np.uint8(imgOrigin)
ky_tu = tv.train_ky_tu(list_img_char)
tv.write_text(img,cnt_plate[stt_cnt],ky_tu)
cv2.imshow('end',imgOrigin)
cv2.waitKey()
'''for img in list_img_char:
    cv2.imshow('img1',img)
    cv2.waitKey()'''
print("License plate has the form : ",ky_tu)
