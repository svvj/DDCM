import numpy as np
import cv2 as cv
import glob

N_corners = (8, 6)      # chess board format
square_size = 25        # mm unit

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((N_corners[1]*N_corners[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:N_corners[0], 0:N_corners[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('data/chess/*.png')


def calibrate_camera():
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, N_corners, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners.reshape(-1, 2), (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv.drawChessboardCorners(img, N_corners, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)

    cv.destroyAllWindows()

    ret, cam_mat, dist_coef, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return objpoints, imgpoints, cam_mat, dist_coef


def solve_pnp(objpoints, imgpoints, mat, d_coef):
    # objps = np.array(objpoints, dtype=float)
    # imgps = np.array(imgpoints, dtype=float).reshape(-1, 2)
    flag = cv.SOLVEPNP_ITERATIVE
    rvecs = []
    tvecs = []
    for i in range(len(images)):
        objps = np.array(objpoints[i], dtype=float)
        imgps = np.array(imgpoints[i], dtype=float).reshape(-1, 2)
        ret, rvec, tvec = cv.solvePnP(objps, imgps, mat, d_coef, flags=flag)
        if ret:
            rvecs.append(rvec)
            tvecs.append(tvec)
    return rvecs, tvecs


if __name__ == "__main__":
    objpoints, imgpoints, cam_mat, dist_coef = calibrate_camera()
    rvec, tvec = solve_pnp(objpoints, imgpoints, cam_mat, dist_coef)
    print(rvec, tvec)
