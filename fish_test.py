import numpy as np
import cv2
import glob
import pickle
import time

# Load previously saved data
with open("fish_parameters.p","rb") as f:
    X = pickle.load(f)
    mtx, dist = [X[i] for i in ('mtx','dist')]

CAM_RES = (1280, 720)
balance = 0.75
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RES[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RES[1])

while True:
    balance = (time.time()/2)%1
    print(balance)
    ret_val, img = cam.read()
    img_dim = img.shape[:2][::-1]  

    #DIM = # dimension of the images used for calibration

    scaled_K = mtx #* img_dim[0] / DIM[0]  
    scaled_K[2][2] = 1.0  
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, dist,
        img_dim, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, dist, np.eye(3),
        new_K, img_dim, cv2.CV_16SC2)
    undist_image = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow('img', undist_image)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        break
