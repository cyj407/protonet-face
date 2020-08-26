import numpy as np
import cv2
import os

path = os.getcwd() + '\\res\\'
savePath = os.getcwd() + '\\data20_me\\'
p_count = 1

def detectFace(frame, saveDir, saveName):
    global p_count
    path = 'D:\\python\\Lib\\site-packages\\cv2\\data\\'
    face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale( frame, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (90, 90))
        if not os.path.exists(savePath + saveDir):
            os.mkdir(savePath + saveDir)
        cv2.imwrite(savePath + saveDir + "\\" + saveName, roi_color)
        p_count = p_count + 1
        break
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


def readDir(k):
    for r, d, f in os.walk(path + k):
        count = 1
        for i in f:
            fpath = os.path.join(r, i)
            dirName = r.split('\\res\\')[1]
            img = cv2.imread(fpath)

            detectFace(img, dirName, str(i))
            count = count + 1


if __name__ == "__main__":
    for i in range(1, 22):
        p_count = 0
        readDir('person' + str(i))
        print('person' + str(i) + ' has {} photos.'.format(p_count))