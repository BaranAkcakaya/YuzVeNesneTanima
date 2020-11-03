import cv2
import imageio

#cv2.namedWindow('a',cv2.WARP_INVERSE_MAP)

face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')


def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #minNeighbors=5 yan yana kac kare tutacagını
    #scaleFactor = 1.3 görüntünün ne kadar küçülecegini
    faces = face.detectMultiScale(gray, 1.3,5)
    i=0
    c='0'
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #r2=cv2.cvtColor(roi_color,cv2.COLOR_BGR2RGB)#resmi renkli yapıyor
        c=str(int(c)+1)
        #cv2.imshow(c,r2)
        eyes = eye.detectMultiScale(roi_gray, 1.1, 7)
        i+=1
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    print(i,'tane yüz var')    
    return frame

image = imageio.imread('2.jpg')
image = detect(frame=image)
imageio.imwrite('output.jpg', image)

