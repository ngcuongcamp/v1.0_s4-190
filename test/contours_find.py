


import cv2
import numpy as np
import imutils
from pyzbar.pyzbar import decode
import zxingcpp 

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
     
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [10, 10],
        [maxWidth - 10, 10],
        [maxWidth - 10, maxHeight - 10],
        [10, maxHeight - 10]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def contours_warped(gray, roi_warped):
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    cv2.imshow('CLOSE 2', closed)
    
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        rect = cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        pts = np.array(box, dtype="float32")
        # print(h,w)
        # if (55 < h < 165) and ( 240 < w < 300): normal
        # if (125 < h < 215) and ( 250 < w < 310): special
        # if (110 < h < 215) and ( 250 < w < 290): 
        
        # nghieng: 107 222 
        # ngang: 56 226
        if (50 < h < 120) and (200 < w < 300):
            # print(h,w)
            cv2.polylines(roi_warped, [pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
            roi_warped = four_point_transform(roi_warped, pts)
    return roi_warped
# cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap2.set(cv2.CAP_PROP_SETTINGS, 1)
# img  = cv2.imread('./test.png')



while(1):
   _, frame2 = cap2.read()
   gray2 =cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
   roi_warped = contours_warped(gray2, frame2)
   
   data_gray = decode(gray2)
#    print('gray ', data_gray)
   cv2.imshow('gray', gray2)
   
#    roi_warped 
   data_wraped = decode(roi_warped)
   cv2.imshow('roi_warped ',roi_warped)
#    print('wraped ', data_wraped)


   _, img_processed = cv2.threshold(cv2.cvtColor(roi_warped, cv2.COLOR_BGR2GRAY), 74, 255, cv2.THRESH_BINARY)
   data_roi_processed = decode(img_processed)
   cv2.imshow('roi_processed', img_processed)
   
   print('wraped ', data_wraped)
   print('roi_processed', data_roi_processed)
   
   if(cv2.waitKey(1) & 0xFF == ord('q')):
      break





