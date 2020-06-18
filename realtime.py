import numpy as np
import cv2
import os
import argparse
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import imutils
from mydataset import MyDataset
from protonet import Protonet
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric


modelFile = "./detect_model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./detect_model/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
path = os.getcwd() + '\\res\\'
savePath = os.getcwd() + '\\data\\'


def markFace(img_path, color):
    frame = cv2.imread(img_path)
    path = 'D:\\python\\Lib\\site-packages\\cv2\\data\\'
    face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale( frame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 5)
        return frame


def detectFace(f_path):
    frame = cv2.imread(f_path)
    path = 'D:\\python\\Lib\\site-packages\\cv2\\data\\'
    face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale( frame, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (90, 90))
        cv2.imwrite("face_" + f_path,roi_color)
        return "face_" + f_path


transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(torch.device('cuda'), torch.float)

def real_time_detect_dnn(frame):
    # process = imutils.resize(frame, width=750)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # sl = int((endY - startY + endX - startX) / 2)
        sl = max(endY - startY, endX - startX)
        startX = startX - 10

        roi_color = frame[startY:startY+sl, startX:startX+sl]
        roi_color = cv2.resize(roi_color, (90, 90))
        
        shot_name = "realtime_save\\face.jpg"
        cv2.imwrite( shot_name, roi_color)
        target = image_loader(shot_name)

        # cv2.rectangle(frame, (startX, startY), (endX, endY),
        cv2.rectangle(frame, (startX, startY), (startX + sl, startY + sl),
            (0, 255, 0), 2)
        
        return target
    return None


def real_time_detect_haar(frame):
    path = 'D:\\python\\Lib\\site-packages\\cv2\\data\\'
    face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale( frame, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (90, 90))
        #### save and read by PIL
        shot_name = "realtime_save\\face.jpg"
        cv2.imwrite( shot_name, roi_color)
        target = image_loader(shot_name)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        return target
    return None


def openCam():
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

    cv2.namedWindow('frame')
    while(cv2.getWindowProperty('frame', 0) >= 0):

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # real_time_detect_haar(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    openCam()