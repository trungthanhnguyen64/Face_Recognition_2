import cv2

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
face_ditector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('Nhap ID tu ban phim:')
print("\n Camera khoi dong")
count=0

while(True):
      ret,img=cam.read()
      img = cv2.flip(img,1)
      gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      face=face_ditector.detectMultiScale(gray,1.3,5)
      for (x,y,w,h) in face:
          cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
          count+=1

          cv2.imwrite('dataSet/User.'+str(face_id)+'.'+str(count)+'.jpg', gray[y:y+h, x:x+w])
          cv2.imshow('image',img)

      k = cv2.waitKey(100) & 0xff
      if k == 27:
         break
      if count >= 50:
         break
print("\n Thoat")
cam.release()
cv2.destroyAllWindows()  # hủy
