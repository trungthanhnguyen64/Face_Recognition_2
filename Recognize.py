import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

font =cv2.FONT_HERSHEY_SIMPLEX
id=0
name =['0','Trung','2','3']
cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
while(True):
      ret,img=cam.read()
      img = cv2.flip(img,1)
      gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      face=faceCascade.detectMultiScale(gray,1.3,5)

      for (x,y,w,h) in face:
          cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

          id,confident = recognizer.predict(gray[y:y+h,x:x+w])
          if (confident < 100):
              id =name[id]
              confident="{0}%".format(round(100-confident))
          else:
              id = "unknown"
              confident = "{0}%".format(round(100 - confident))
          cv2.putText(img,str(id),(x+5,y-5),font,1,(255,255,255),2)
          cv2.putText(img, str(confident), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

      cv2.imshow('Nhan dien khuon mat',img)
      k = cv2.waitKey(100) & 0xff

      if k == 27:
          break

print("\n Hoan Thanh")
cam.release()
cv2.destroyAllWindows()  # hủy

