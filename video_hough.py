import cv2 as cv
import numpy as np

cap = cv.VideoCapture('60fps_DDCM_harder.mp4')
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv.CAP_PROP_FPS)
print(f"width: {width}, height: {height}, fps: {fps}")
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('hough_harder.mp4', fourcc, fps, (int(width), int(height)))


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

    circles = cv.HoughCircles(thresh, cv.HOUGH_GRADIENT, 1, 1, param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for i in circles:
            cv.circle(thresh, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)


    # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # with_contours = cv.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # Show keypoints
    out.write(frame)
    cv.imshow("contours", thresh)

    if cv.waitKey(1) == ord('q'):
        break

# cv.imshow('img', frame)
cv.waitKey(0)
cv.destroyAllWindows()