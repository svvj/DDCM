# Capture Video from Camera

import numpy as np
import cv2 as cv

cap = cv.VideoCapture('data/slow.mp4')
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv.CAP_PROP_FPS)
print(f"width: {width}, height: {height}, fps: {fps}")
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('slow_contours_momentum.mp4', fourcc, fps, (int(width), int(height)))

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Display the resulting frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    potential_marker = []

    for i in contours:
        M = cv.moments(i)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            potential_marker.append([cX, cY])

            cv.circle(frame, (cX, cY), 3, (255, 0, 0), -1)
            # cv.drawContours(frame, [i], 0, (0, 0, 255), 1)
        else:
            print(f"PassingZeroDivision: M['m00'] is zero in {i}")

    # Show keypoints
    out.write(frame)
    # cv.imshow("contours", with_contours)

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv.destroyAllWindows()
