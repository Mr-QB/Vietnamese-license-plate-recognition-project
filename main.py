import cv2
import thuvien as tv 
import numpy as np

#plate
img = cv2.imread('D:/test/test.png')#đọc ảnh
contour = tv.tien_su_ly(img) # rút ra contour của toàn ảnh
list_img_plate,cnt_plate = tv.detect_plate(contour,img) #rút ra ảnh có khả năng là plate 
#char
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
print("biển số xe có dạng : ",ky_tu)