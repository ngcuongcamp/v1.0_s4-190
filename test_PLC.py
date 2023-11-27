import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QDesktopWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from Ui_design import Ui_MainWindow
import logging
import serial
import datetime
import shutil
import cv2
import glob
import time
import os
import numpy as np
from pyzbar.pyzbar import decode
import zxingcpp
import imutils
import configparser
from pygrabber.dshow_graph import FilterGraph

# initial variables
path_dir_log = "./logs/"
time_day = time.strftime("%Y_%m_%d")

# logger handler
# def setup_logger():
#     for file_name in glob.glob(path_dir_log + '/*.log'):
#         if time_day not in file_name:
#             os.remove(file_name)
#             print("delete file:", file_name)

#     logger = logging.getLogger('MyLogger')
#     logger.setLevel(logging.DEBUG)  
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     file_handler = logging.FileHandler(path_dir_log + time_day + '.log')
#     file_handler.setFormatter(formatter)
#     file_handler.setLevel(logging.DEBUG)
#     logger.addHandler(file_handler)
#     return logger
# logger = setup_logger()


# timer handler
# def get_current_date(): 
#    return datetime.datetime.now().strftime("%Y-%m-%d")

# manage folder 
def create_daily_folders():
   config = configparser.ConfigParser()
   config.read('./config.ini')
   path = config['PATH']['IMAGE_NG_FOLDER']
   current_date = datetime.datetime.now()
   folder_name = current_date.strftime("%Y-%m-%d")
   folder_path = os.path.join(path, folder_name)

   if not os.path.exists(folder_path):
      os.makedirs(folder_path)
      print(f"Created new folder: {folder_path}")

def handle_remove_old_folders():
   config = configparser.ConfigParser()
   config.read('./config.ini')
   folder_to_keep = int(config['SETTING']['FOLDER_TO_KEEP'])
   path = config['PATH']['IMAGE_NG_FOLDER']
   
   subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
   subfolders.sort()
   if len(subfolders) > folder_to_keep:
      folders_to_delete = subfolders[:len(subfolders) - folder_to_keep]
      for folder_to_delete in folders_to_delete:
         try:
            shutil.rmtree(folder_to_delete)
            print(f"Removed old folder: {folder_to_delete}")
         except Exception as e:
            print(f"Remove error '{folder_to_delete}': {e}")
handle_remove_old_folders()
create_daily_folders()


# PLC THREAD
class PLCThread(QThread):
   data_received = pyqtSignal(bytes)
   signal_error = pyqtSignal()
   
   def __init__(self, port, baudrate, timeout=0.009):
      super(PLCThread, self).__init__()
      self.port = port 
      self.baudrate = baudrate
      self.timeout = timeout
      self.is_running = False
      
   def connect_serial(self):
      try:
         self.serial_port = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
         )
         self.is_running = True
      except Exception as e:
         print(e, 'Connect PLC Error')
   
   def send_signal_to_plc(self, data:str):
      if self.serial_port and self.serial_port.is_open:
         self.serial_port.write(data.strip())
   
   def run (self):
      try:
         self.connect_serial()
         while self.is_running:
            if self.serial_port and self.serial_port.is_open:
               data = self.serial_port.readline()
               if len(data) > 0:
                  MyApplication.flag_signal_plc = True
                  if MyApplication.flag_signal_plc:  
                     self.data_received.emit(data)
                     MyApplication.flag_signal_plc = False
      
      except Exception as e:
         self.signal_error.emit()
         # logger.error(f'Connect PLC Error: {e}')   
   def stop(self):
      self.is_running = False
      if self.serial_port and self.serial_port.is_open:
         self.serial_port.close()


# SFC THREAD
# class SFCThread(QThread):
#    data_received = pyqtSignal(bytes)
#    signal_error = pyqtSignal()
   
#    def __init__(self, port, baudrate, timeout=0.009):
#       super(SFCThread, self).__init__()
#       self.port = port 
#       self.baudrate = baudrate
#       self.timeout = timeout
#       self.is_running = False
   
#    def connect_serial_sfc(self):
#       self.serial_sfc = serial.Serial(
#          port=self.port,
#          baudrate=self.baudrate,
#          bytesize=serial.EIGHTBITS,
#          parity=serial.PARITY_NONE,
#          stopbits=serial.STOPBITS_ONE
#       )
#       self.is_running = True
   
#    def send_signal_to_sfc(self, data:str):
#       if self.serial_sfc and self.serial_sfc.is_open:
#          self.serial_sfc.write(data.strip())
         
#    def run(self):
#       try:
#          self.connect_serial_sfc()
#          while self.is_running:
#             if self.serial_sfc and self.serial_sfc.is_open:
#                data = self.serial_sfc.read()
#                if len(data) > 0:
#                   MyApplication.flag_signal_sfc = True
#                   if MyApplication.flag_signal_sfc:  
#                      self.data_received.emit(data)
#                      MyApplication.flag_signal_sfc = False
#       except Exception as e:
#          self.signal_error.emit()
#          # logger.error(f'Connect SFC Error: {e}')

#    def stop(self):
#       self.is_running = False
#       if self.serial_sfc and self.serial_sfc.is_open:
#          self.serial_sfc.close()
   
# CAMERA THREAD
class CameraThread(QThread):
   frame_received = pyqtSignal(np.ndarray)
   update_error_signal = pyqtSignal()
   
   def __init__(self, camera_id):
      super(CameraThread, self).__init__()
      self.camera_id = camera_id
      self.is_running = True
      self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
      
   def run(self):
      while self.is_running:
         try:
            count_input_devices = len(FilterGraph().get_input_devices())
            if count_input_devices == 1 : 
               self.ret, self.frame = self.cap.read()
               if self.ret:
                  self.frame_received.emit(self.frame)
            elif count_input_devices < 1:
               self.update_error_signal.emit()
               # stop thread
               self.stop() 
               self.wait() 
               
         except Exception as e:
            time.sleep(0.1)
            self.cap.release()
            # logger.error(f'Camera {self.camera_id} error: {e}')
         cv2.waitKey(1)
         
   def stop(self):
      self.is_running = False
      self.requestInterruption()
      self.cap.release()
      self.quit()
      
   

class MyApplication(QMainWindow):
   frame1 = None 
   data_scan1 = None
   data_scan2 = None
   count_frame = 0
   flag_signal_plc = False
   # flag_signal_sfc = False
   is_finished = [False, False]
   
   def __init__(self):
      super().__init__()
      
      # initialize UI
      self.initial_UI_MainWindow()
      
      # read config 
      self.config = configparser.ConfigParser()
      self.config.read('./config.ini')
      self.read_config()
      self.is_update_cam_error = True
      
      
      # thread camera 
      self.THREAD_CAMERA_1 = CameraThread(self.ID_C1)
      self.THREAD_CAMERA_1.frame_received.connect(self.display_frame1)
      self.THREAD_CAMERA_1.start()
      if self.IS_OPEN_CAM_PROPS == 1:
         self.THREAD_CAMERA_1.cap.set(cv2.CAP_PROP_SETTINGS, 1)
      self.THREAD_CAMERA_1.update_error_signal.connect(self.update_status_camera_error)
      
      
      # thread PLC (S4-200)
      self.THREAD_PLC = PLCThread(self.COM_PLC, self.BAUDRATE_PLC)
      self.THREAD_PLC.start()
      # The handle_signal_plc function is called when a signal is received from the PLC.
      self.THREAD_PLC.data_received.connect(self.handle_signal_plc)
      
      # thread SFC
      # self.THREAD_SFC = SFCThread(self.COM_SFC, self.BAUDRATE_SFC)
      # self.THREAD_SFC.start()
      # self.THREAD_SFC.data_received.connect(self.handle_signal_sfc)
      
      
   # initial background frame  
   def initial_UI_MainWindow(self):
      self.Uic = Ui_MainWindow()
      self.Uic.setupUi(self)
      self.setWindowIcon(QIcon('./icons/Logo.ico'))
      # set position window
      screen_geometry = QDesktopWidget().availableGeometry()
      self.setGeometry(screen_geometry.width() - self.width(), screen_geometry.height() - self.height(), self.width(), self.height())
      
      # origin background
      original_pixmap = QPixmap('./icons/bg-no-camera.png')
      scaled_pixmap = original_pixmap.scaled(self.Uic.Frame1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
      self.Uic.Frame1.setPixmap(scaled_pixmap)
      self.show()
   
      
   # read config handler
   def read_config(self):
      self.COM_PLC = self.config["PLC"]["COM"]
      self.BAUDRATE_PLC = self.config["PLC"]["BAUDRATE"]
      self.COM_SFC = self.config["SFC"]["COM"]
      self.BAUDRATE_SFC = self.config["SFC"]["BAUDRATE"]
      self.ID_C1 = int(self.config["CAMERA"]['IDC1'])
      self.SCAN_LIMIT = int(self.config["SETTING"]['SCAN_LIMIT'])
      self.IS_OPEN_CAM_PROPS = int(self.config["SETTING"]['IS_OPEN_CAM_PROPS']) 
   
   def format_current_time(self): 
      current_time = datetime.datetime.now()
      formatted_time = current_time.strftime("%Y-%m-%d %H-%M-%S")
      return formatted_time
   
   # display frame
   def display_frame1(self,frame):
      self.frame1 = frame
      self.gray1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2GRAY)
      
      frame_rgb = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
      img = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
      scaled_pixmap = img.scaled(self.Uic.Frame1.size())
      pixmap = QPixmap.fromImage(scaled_pixmap)
      self.Uic.Frame1.setPixmap(pixmap)

   # custom close application
   def closeEvent(self, event):
      req = QMessageBox.question(self, 'Confirm Close', 'Do you want to close the application?',QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
      if req == QMessageBox.Yes:
         PLCThread.send_signal_to_plc(b'5/r/n')
         event.accept()  
      else:
         event.ignore()  

   def update_status_camera_error(self):
      self.is_update_cam_error = True
      if self.is_update_cam_error: 
         # logger.error(f'CAM ERROR')
         self.is_update_cam_error = False
   
         
      self.Uic.ResultContent.setText('CAM ERROR')
      self.Uic.ResultContent.setStyleSheet('font-size: 16px;\nfont: 16pt "Segoe UI"; border: 1px solid #ccc; color: #fff; background-color: #a84632;')
      original_pixmap = QPixmap('./icons/bg-no-camera.png')
      scaled_pixmap = original_pixmap.scaled(self.Uic.Frame1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
      self.Uic.Frame1.setPixmap(scaled_pixmap)
      
   
   # handle plc signal
   def handle_signal_plc(self, data):
      print('Nhan tin hieu tu S4-200:', data)
      
      # error S4-200 scan QRCode -> set state to default
      if data == b'0' and self.frame1 is not None: 
         print(f'Failed to scan QRCODE (S4-200): {data}')
         # logger.error(f'Failed to scan QRCode (S4-200): {data}')
         
         self.data_scan1 = None 
         self.data_scan2 = None
         self.is_finished = [False, False]
         self.count_frame = 0
         self.Uic.ResultContent.setText('FAIL S4-200')
         self.Uic.ResultContent.setStyleSheet('font-size: 16px;\nfont: 16pt "Segoe UI"; border: 1px solid #ccc; color: #fff; background-color: #a84632;')
      
      # request scan 3Code from S4-200 
      if data == b'1' and self.frame1 is not None:
         print(f'Nhan tin hieu scan tu PLC: {data}')
         # logger.info(f'Nhan tin hieu scan tu PLC: {data}')
         
         self.data_scan1 = None 
         while self.count_frame < self.SCAN_LIMIT:
            self.count_frame +=1
            if self.data_scan1 is None: 
               self.data_scan1 = self.read_barcode_zxingcpp(self.frame1, self.gray1)
               self.data_scan1 = b'test'
            if self.data_scan1 != None:
               break
            
            if self.count_frame >= self.SCAN_LIMIT:
               if self.data_scan1 is None: 
                  break
         self.is_finished[1] = True
      
      # request reset program from S4-200
      if data == b'5' and self.frame1 is not None: 
         print(f'Nhan tin hieu reset tu PLC: {data}')
         # logger.info(f'Nhan tin hieu reset tu PLC: {data}')
         
         self.Uic.ResultContent.setStyleSheet('font-size: 16px;\nfont: 16pt "Segoe UI"; border: 1px solid #ccc; color: #fff; background-color: #a2a832;')
         self.Uic.ResultContent.setText('RESET')
         self.is_finished = [False, False]
         self.data_scan1 = None
         self.data_scan2 = None
         self.count_frame = 0
      
      # request close program from S4-200
      if data ==b'9' and self.frame1 is not None :
         print(f'Co tin hieu dong chuong trinh tu PLC: {data}')
         # logger.info(f'Co tin hieu dong chuong trinh tu PLC :{data}')
         # self.THREAD_PLC.send_signal_to_plc(b'9\r\n')
         time.sleep(0.5)
         self.closeEvent()  
         
         
      # respone (QRCode data) from S4-200
      if len(data) > 5 and self.frame1 is not None: 
         # save Q to var
         self.data_scan2 = data
         self.is_finished[0] = True
      
      # conditions checker
      # scanned QRCode(S4-200) and 3Code(S4-190)
      if self.is_finished[0] is True and self.is_finished[1] is True:
         # PASS SCAN
         if self.data_scan1 is not None and self.data_scan2 is not None: 
            print(f'Scanned OK\ndata_cam1: {self.data_scan1} - data_cam2: {self.data_scan2}')
            # logger.info(f'Scanned OK\ndata_cam1: {self.data_scan1} - data_cam2: {self.data_scan2}')
            
            # self.THREAD_SFC.send_signal_to_sfc(self.data_scan1)
            # time.sleep(0.4)
            # self.THREAD_SFC.send_signal_to_sfc(self.data_scan2)   
            self.THREAD_PLC.send_signal_to_plc(b'01')

            self.count_frame = 0
            self.Uic.ResultContent.setText('PASS SCAN')
            self.Uic.ResultContent.setStyleSheet('font-size: 16px;\nfont: 16pt "Segoe UI"; border: 1px solid #ccc; color: #fff; background-color: #32a851;')

         
         # FAIL SCAN
         if self.data_scan1 is None: 
            print(f'Failed to scan Barcode(S4-190) \ndata_cam1: {self.data_scan1} - data_cam2: {self.data_scan2}')
            # logger.error(f'Failed to scan Barcode(S4-190) \ndata_cam1: {self.data_scan1} - data_cam2: {self.data_scan2}')
            
            # save image error
            # image_filename = "image_NG/{}/{}.png".format(get_current_date(),self.format_current_time())
            # cv2.imwrite(image_filename, self.frame1)
            
            
            self.THREAD_PLC.send_signal_to_plc(b'0\r\n')
            self.Uic.ResultContent.setText('FAILED SCAN')
            self.Uic.ResultContent.setStyleSheet('font-size: 16px;\nfont: 16pt "Segoe UI"; border: 1px solid #ccc; color: #fff; background-color: #a84632;')
         
         self.is_finished = [False, False]
         self.data_scan1 = None
         self.data_scan2 = None
         self.count_frame = 0

      # refresh state UI
      if self.is_finished == [False, True] or self.is_finished == [True, False]:
         self.Uic.ResultContent.setText('NONE')
         self.Uic.ResultContent.setStyleSheet('font-size: 16px;\nfont: 16pt "Segoe UI"; border: 1px solid #ccc; color: #999; background-color: #fff;')
      
      
   # def handle_signal_sfc(self, data):
   #    # if data == b'0' or data == b"\x00\x00":
   #    if data == b'00' and self.frame1 is not None:
   #       print('Tin hieu FAIL SFC')
   #       self.Uic.ResultContent.setText('FAILED')
   #       self.Uic.ResultContent.setStyleSheet('font-size: 16px;\nfont: 16pt "Segoe UI"; border: 1px solid #ccc; color: #fff; background-color: #a84632;')
   #       # logger.error(f'Nhan tin hieu FAIL tu SFC: {data}')
         
   #       self.THREAD_PLC.send_signal_to_plc(b'00\r\n')
   #       # logger.info('Gui tin hieu FAIL SFC check cho PLC')
         
   #    # elif data == b'1' or data == b'\x01\x00':
   #    elif data == b'01' or data == b'\x01\x00':
   #       print('Tin hieu OK SFC')
   #       self.Uic.ResultContent.setText('PASS')
   #       self.Uic.ResultContent.setStyleSheet('font-size: 16px;\nfont: 16pt "Segoe UI"; border: 1px solid #ccc; color: #fff; background-color: #32a851;')
   #       # logger.info(f'Nhan tin hieu PASS check tu SFC: {data}')
   #       self.THREAD_PLC.send_signal_to_plc(b'01\r\n')
   #    else:
   #       print(f'SFC gui sai tin hieu: {data}')
   #       # logger.warning(f'SFC gui sai tin hieu: {data}')
      
   # process image 
   def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
     
   def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
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
   
   def contours_warped(self, gray, roi_warped):
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
        # cv2.imshow("close", closed)

        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i in contours:
            x, y, w, h = cv2.boundingRect(i)
            rect = cv2.minAreaRect(i)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            pts = np.array(box, dtype="float32")

            # if (55 < h < 165) and ( 240 < w < 300): normal
            # if (125 < h < 215) and ( 250 < w < 310): sp
            # if (80 < h < 80) and (200 < w < 500): old
            # if (80 < h < 250) and (200 < w < 500): 
            if (80 < h < 80) and ( 200 < w < 500):
                # print(h, w)
                roi = roi_warped[y:y + h, x:x + w].copy()
                roi_warped = self.four_point_transform(roi_warped, pts)
        return roi_warped
   
   
   # process read code 
   def read_barcode_zxingcpp(self, frame, gray):
        roi_warped = self.contours_warped(gray, frame)
        roi_processed = cv2.cvtColor(roi_warped, cv2.COLOR_BGR2GRAY)
        _,roi_processed = cv2.threshold(roi_processed, 74, 255, cv2.THRESH_BINARY)
      #   cv2.imshow('roi_processed', roi_processed)
        
        data = zxingcpp.read_barcode(frame)
        if data is not None and len(data.text) == 11:
            return data.text.encode('utf-8')
        else:
            data = zxingcpp.read_barcode(gray)
            if data is not None and len(data.text) == 11:
                return data.text.encode('utf-8')
            else:
                data = zxingcpp.read_barcode(roi_warped)
                if data is not None and len(data.text) == 11:
                    return data.text.encode('utf-8')
                else:
                    data = zxingcpp.read_barcode(roi_processed)
                    if data is not None and len(data.text) == 11:
                        return data.text.encode('utf-8')
                    else:
                        return self.read_barcode_pyzbar(roi_warped,roi_processed, frame, gray)
   def read_barcode_pyzbar(self, roi_warped, roi_processed, frame, gray):
        data = decode(frame)
        if len(data) > 0 and len(data[0].data) == 11: 
            return data[0].data
        else:
            data = decode(gray)
            if len(data) > 0 and len(data[0].data) == 11: 
                return data[0].data
            else:
                data = decode(roi_warped)
                if len(data) > 0 and len(data[0].data) == 11: 
                    return data[0].data
                else:
                    data = decode(roi_processed)
                    if len(data) > 0 and len(data[0].data) == 11:
                        return data[0].data
                    else:
                        return None
      
   
if __name__ == '__main__':
   app = QApplication(sys.argv)
   window = MyApplication()
   sys.exit(app.exec_())