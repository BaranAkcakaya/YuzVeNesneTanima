import cv2
import os
import numpy as np
import json
from PIL import Image

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

#Verinin toplandığı bölüm
user = input("Lütfen Kullanici Adi Giriniz: ").lower()
print("[Warning] KAMERAYA BAKIN VE BEKLEYIN..")
if('dataset' not in str(os.listdir())):
    # os.mkdir("dataset/" + user.lower())       #Böyle yapınca cok hata veriyo
    os.mkdir('dataset')
if(str(user) not in str(os.listdir('dataset'))):
    os.mkdir('dataset/'+user)
temp = 0

# while(True):
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(gray, 1.5, 5)
    
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         temp += 1
#         print(temp)
#         cv2.imwrite("dataset/"+ user + '/' + str(temp+1) + ".jpg", gray[y:y+h,x:x+w])        
#         cv2.imshow('frame',frame)        
#     if cv2.waitKey(1) & 0xFF == ord('Q'):
#         break
#     elif(temp > 100):
#         break
    
cap.release()
cv2.destroyAllWindows()

#Egitildiği bölüm
path = 'dataset'
lbp = cv2.face.LBPHFaceRecognizer_create()

def getImagesLabels(path):
    faceSamples = []
    ids = []
    files = os.listdir(path)
    dictinary = {}
    for i, fl in enumerate(files):
        dictinary[fl] = int(i)
    file = open('ids.json', 'w')
    tempA = json.dump(dictinary, file)
    file.close()
    
    for fl in files:
        for img in os.listdir(os.path.join(path, fl)):
            pilImg = Image.open(os.path.join(path, fl, img)).convert("L")
            imgNumpy = np.array(pilImg, 'uint8')
            id = int(dictinary[fl])
            faces = face_detector.detectMultiScale(imgNumpy)
            for (x, y, w, h) in faces:
                faceSamples.append(imgNumpy[y:y+h, x:x+w])
                ids.append(id)
    
    return faceSamples, ids

print("Veri Egitiliyor")
faces, ids = getImagesLabels("dataset")
print("Veri Egitildi")
lbp.train(faces, np.array(ids))
lbp.write('trainer.yml')


#Burası sonuc bölümü
lbp.read('trainer.yml')
font = cv2.FONT_HERSHEY_SIMPLEX
tempId = 0
tempDictionary = {}
names = []
tempFile = open("ids.json", "r")
tempDictionary = json.load(tempFile)

for key, value in tempDictionary.items():
    names.append(key)

cap2 = cv2.VideoCapture(0)

while(True):
    ret, frame = cap2.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.5, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        tempNu, tempO = lbp.predict(gray[y:y+h,x:x+w])
        print(tempNu)
        
        if(tempO < 70):
            tempNu = names[tempNu]
        else:
            tempNu = 'Unknown'
    
        cv2.putText(frame, str(tempNu), (x +5, y - 5), font, 1, (255, 255, 255), 2)
        
    cv2.imshow('CAMERA', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap2.release()
cv2.destroyAllWindows()