

import cv2
import numpy as np
from PIL import Image
import os

path ='dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create() #Dòng code trên tạo ra một đối tượng recognizer của lớp cv2.face.LBPHFaceRecognizer. Lớp này là một trong những lớp recognizer được sử dụng phổ biến trong nhận dạng khuôn mặt sử dụng OpenCV và hỗ trợ việc huấn luyện các mô hình nhận dạng khuôn mặt dựa trên phương pháp nhận dạng khuôn mặt địa phương nhị phân (Local Binary Patterns Histograms Face Recognizer).
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImageAndLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]# Dòng code trên tạo ra một danh sách imagePaths bằng cách lặp qua tất cả các tệp trong thư mục được chỉ định bởi biến path, sử dụng hàm os.listdir(path) để lấy danh sách tất cả các tệp trong thư mục đó. Với mỗi tệp, hàm os.path.join(path, f) được sử dụng để ghép đường dẫn đầy đủ đến tệp đó với đường dẫn của thư mục chứa tệp (path), sau đó đưa đường dẫn đầy đủ này vào danh sách imagePaths.

    faceSample = []
    ids = []

    for imagePath in imagePaths:
        PIL_image = Image.open(imagePath).convert('L') #Chuyển sang ảnh xám bằng convert('L')
        img_numpy = np.array(PIL_image, 'uint8')#ảnh xám được chuyển đổi thành một mảng numpy (ndarray) với kiểu dữ liệu uint8 (unsigned integer 8-bit), trong đó mỗi giá trị của mảng tương ứng với một pixel trong ảnh.

        id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces=detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSample.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSample, ids
print('\n Đang train')

faces, ids = getImageAndLabel(path)
recognizer.train(faces,np.array(ids))

recognizer.write('trainer/trainer.yml')
print("Hoan Thanh Train")
print("Co {0} khuon mat duoc train.".format(len(np.unique(ids))))