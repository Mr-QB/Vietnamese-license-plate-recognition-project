import cv2
import numpy as np
import os
folder1 = 'D:\Code python\Lam-quen-voi-thu-vien-OpenCV/train1'
folder2 = 'D:\Code python\Lam-quen-voi-thu-vien-OpenCV/train2'

w,h=20,20

def tien_su_ly(img): #Hàm trả về list các contour của ảnh có thể là ký tự
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    noise_removal = cv2.bilateralFilter(img_gray,9,75,75)#bộ lọc làm mờ 
    equal_histogram = cv2.equalizeHist(noise_removal)#cân bằng lại hình ảnh, làm ảnh không quá sáng hoặc tối
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))#lấy kernel
    morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=20)#
    sub_morp_image = cv2.subtract(equal_histogram,morph_image)#xóa phông, kết hợp giữa ảnh đã làm mờ và ảnh đã lọc ra hình góc(morphologyEx) trái ngược với cv2.add
    ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)#ngưỡng ảnh
    canny_image = cv2.Canny(thresh_image,250,255)#lấy egde
    kernel = np.ones((3,3), np.uint8)
    dilated_image = cv2.dilate(canny_image,kernel,iterations=1)#tăng độ sắc nhọn cho egde
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#lấy contours
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]#lấy ra 10 contour có diện tích lớn nhất
    return contours #trả về list 10 contours có diện tích lớn nhất

def wh(arr):
    min_x=1000000
    min_y=1000000
    max_x=0
    max_y=0
    for a in arr:
        if(a[0]>max_x):
            max_x=a[0]
        if(a[1]>max_y):
            max_y=a[1]
        if(a[0]<min_x):
            min_x=a[0]
        if(a[1]<min_y):
            min_y=a[1]
    
    return min_x,max_x,min_y,max_y

def cut(img,x_min,x_max,y_min,y_max):
    img_cut = img[y_min:y_max,x_min:x_max]
    return img_cut

def detect_plate(contours,img):
    screenCnt = []
    w_img,h_img,_=img.shape
    list_img_filter_plate = []
    for c in contours:
        _,_,w,h = cv2.boundingRect(c)
        peri = cv2.arcLength(c, True)#lấy ra chu vi của từng contour
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)#xấp xỉ đa giác
        if len(approx) == 4 and w/w_img>0.01 and h/h_img>0.01 :#nếu là tứ giác
            approx=approx.reshape(-1,2)
            screenCnt.append(approx)
    a = screenCnt
    for ai in a:
        x_min,x_max,y_min,y_max = wh(ai)
        list_img_filter_plate.append(cut(img,x_min,x_max+3,y_min,y_max+3))
    return list_img_filter_plate,a

def check(contours,img): #kiểm tra đối tượng nào có thể là ký tự   
    contour_filter = []
    sum = 0           
    area_cnt = [cv2.contourArea(cnt) for cnt in contours]#tạo mảng lưu diện tích các contours
    area_sort = np.argsort(area_cnt)[::-1] #sắp xếp thứ tự các contours theo diện tích giảm dần
    area_sort[:20]#lấy 20 phần tử có diện tích lớn nhất
    w_img,h_img,_=img.shape
    for a in area_sort[:10]:
        cnt = contours[a]
        sum = 0
        for b in area_sort[:10]:
            cnt_i = contours[b]    
            _,_,w,h = cv2.boundingRect(cnt)
            _,_,w_i,h_i = cv2.boundingRect(cnt_i)
            if abs(w-w_i)<5 and abs(h-h_i)<5 and w/w_img>0.1 and h/h_img>0.1:
                sum+=1
            if sum >= 2:
                contour_filter.append(cnt)
                sum=0
                break
    return contour_filter

def take_second(elem):
    return elem[0][0][0]  

def detect_char(list_img_filter_plate): #lọc các contours, xem vùng nào có khả năng là ký tự
    len_contour_fit=0
    img_filter_plate_fit=[]
    contour_char_fit = []
    list_img_char_fit = []
    stt_cnt = -1
    stt_cnt_end = None
    for img_filter_plate in list_img_filter_plate:
        roi_gray = cv2.cvtColor(img_filter_plate,cv2.COLOR_BGR2GRAY)
        roi_blur = cv2.GaussianBlur(roi_gray,(3,3),1)
        thre = cv2.adaptiveThreshold(roi_blur,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 21)#ngưỡng ảnh
        #cv2.imshow('end',thre)
        #cv2.waitKey()
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        #thre_mor = cv2.morphologyEx(thre,cv2.MORPH_DILATE,kerel3)
        contours,hier = cv2.findContours(thre,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)    
        contour_char = check(contours,img_filter_plate)
        contour_char = sorted(contour_char, key=take_second)#sắp xếp lại contour theo thứ tự trái qua phải
        img_char = cut_img(img_filter_plate,contour_char)
        stt_cnt+=1
        if len(contour_char)>len_contour_fit  :
            len_contour_fit = len(contour_char)
            img_filter_plate_fit = img_filter_plate
            contour_char_fit = contour_char
            list_img_char_fit = img_char
            stt_cnt_end = stt_cnt
    return img_filter_plate_fit,contour_char_fit,list_img_char_fit,stt_cnt_end

def cut_img(img,contour_filter): #hàm lọc ra các ảnh đối tượng nhỏ để xét, thay vì xét trên toàn bộ img ban đầu
    img_filter = []
    for cnt in contour_filter:
        img_1 = img.copy()
        x,y,w,h = cv2.boundingRect(cnt) 
        img_cut = img[y-3:y+h+3,x-3:x+w+3]
        img_filter.append(img_cut)
    return img_filter

def train_ky_tu(list_img_char):#dùng knn để nhận biết ký tự có trong biển số
    strChars = ''
    data_train = np.loadtxt("data_train.txt", np.float32)
    data_train_labels = np.loadtxt("data_train_labels.txt", np.float32)
    for img in list_img_char:
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgROIResized = cv2.resize(img_gray, (w,h))
        npaROIResized = imgROIResized.reshape((1, w*h))
        npaROIResized = np.float32(npaROIResized)
        kNearest = cv2.ml.KNearest_create()
        kNearest.setDefaultK(1)                                                         
        kNearest.train(data_train, cv2.ml.ROW_SAMPLE, data_train_labels) 
        _,kq,_,_ = kNearest.findNearest(npaROIResized,3)
        strCurrentChar = str(chr(int(kq[0][0])))
        strChars = strChars + strCurrentChar 
    return strChars

def drawBoundingBox(img, cnt):
    x,y,w,h = cv2.boundingRect(cnt)#tìm ra tọa độ điểm góc phía bên trái , chiều cao, chiều rộng của 1 contour 
    img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)#vẽ các hình vuông bao quanh contour theo thống số tính được phía trên
    return img

def write_text(img,cnt,strChars):
    x,y,_,_ = cv2.boundingRect(cnt)
    cv2.putText(img,strChars,(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    