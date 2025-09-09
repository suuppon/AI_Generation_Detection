import cv2
import os 

folder = "/home/jwchoi/project/jskim/images/fake/face_video_1148.mp4"


files = sorted(os.listdir(folder))
for f in files[:5] + files[-5:]:
    path = os.path.join(folder, f)
    img = cv2.imread(path)
    if img is not None:
        print(f, img.shape)
    else:
        print(f, "읽기 실패")