import numpy as np
import cv2
import os

path = os.getcwd() + '\\res\\'
savePath = os.getcwd() + '\\data\\'

def detectFace(frame):
    path = 'D:\\python\\Lib\\site-packages\\cv2\\data\\'
    face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale( frame, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (150, 150))
        break
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # DNN = "TF"
    # if DNN == "CAFFE":
    #     modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    #     configFile = "deploy.prototxt"
    #     net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    # else:
    #     modelFile = "opencv_face_detector_uint8.pb"
    #     configFile = "opencv_face_detector.pbtxt"
    #     net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)


def openCam():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

    cv2.namedWindow('frame')
    # cv2.resizeWindow('frame', 960, 720)

    while(cv2.getWindowProperty('frame', 0) >= 0):

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
            
        detectFace(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize():
    pass

if __name__ == "__main__":
    openCam()