import numpy as np
import cv2
import glob
import pickle

CAM_RES = (1280, 720)

# Checkboard dimensions
CHECKERBOARD = (6,9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)# Arrays to store object points and image points from all the images.

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RES[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RES[1])

IMG_AMT = 20
while len(objpoints) < IMG_AMT:
 
    ret_val, img = cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners2)
        print(len(objpoints),IMG_AMT)

        cv2.imwrite(f"calib_imgs/chess-{len(objpoints):03}.png", img)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners2, ret)
    cv2.imshow('img', img)
    cv2.waitKey(500)
    
cv2.destroyAllWindows()

# calculate K & D
N_imm = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

pickle_obj = {"mtx":mtx,"dist":dist,"RMS":ret}

with open("fish_parameters.p", "wb") as f:
    pickle.dump(pickle_obj, f)

print("RMS:", ret)

