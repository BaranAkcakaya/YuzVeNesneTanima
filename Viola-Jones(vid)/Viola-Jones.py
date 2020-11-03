# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:54:36 2020

@author: lenovoz
"""

import cv2
import imageio

#cv2.namedWindow('resim',cv2.WINDOW_NORMAL)
face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

#video icindeki tek bir görüntü karesine frame denir
def detect(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #rectangle=dikdörtgen
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        gray_face = gray[y:y+h,x:x+w]
        color_face = frame[y:y+h,x:x+w]
        #c2=cv2.cvtColor(color_face,cv2.COLOR_BGR2RGB)
        #cv2.imshow('resim',c2)
        eyes = eye.detectMultiScale(gray_face,1.1,3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(color_face,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    return frame

reader=imageio.get_reader('1.mp4')
fps=reader.get_meta_data()['fps']
writer=imageio.get_writer('new.mp4',fps=fps)

for i,frame in enumerate(reader):
    frame=detect(frame)
    writer.append_data(frame)
    print(i)
writer.close()