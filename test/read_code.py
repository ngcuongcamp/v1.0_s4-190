import cv2 
from pyzbar.pyzbar import decode
import zxingcpp



cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
img  = cv2.imread('./test.png')


while True: 
   _, frame = cap.read()
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   cv2.imshow('img', img)

   
   data = decode(img)
   data2 = zxingcpp.read_barcode(img)
   print(data, 'data')
   print(data2, "data2")
   
   if(cv2.waitKey(1) & 0xFF == ord('q')):
      break
   
cv2.destroyAllWindows()
cap.release()

# // 11