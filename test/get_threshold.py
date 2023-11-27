import cv2
import numpy as np

def callback(x):
   pass
# cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(0)
cv2.namedWindow('image')

cv2.createTrackbar('T', 'image', 0, 255, callback)
img  = cv2.imread('./test.png')


while(1):
   _, frame2 = cap2.read()
   # frame2 = img
   
   T = cv2.getTrackbarPos('T','image')
   
   # gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
   
   # _,Binary = cv2.threshold(gray, T, 255,cv2.THRESH_BINARY)
   _,B2 = cv2.threshold(gray2, T, 255, cv2.THRESH_BINARY)
   
   
   # cv2.imshow('CAM 1', Binary)
   cv2.imshow('CAM 2', B2)
   if(cv2.waitKey(1) & 0xFF == ord('q')):
      break


cv2.destroyAllWindows()
cap2.release()