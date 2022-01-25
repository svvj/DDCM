# Identify each markers

import numpy as np
import cv2 as cv
import hash

cap = cv.VideoCapture('240fps.mp4')
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
section_num = 6
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width_section = width / section_num
height_section = height / section_num

fps = cap.get(cv.CAP_PROP_FPS)
print(f"width: {width}, height: {height}, fps: {fps}")
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('video_marker_240.mp4', fourcc, fps, (int(width), int(height)))

save_video = False
visual_video = True

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 110, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    # with_contours = cv.drawContours(frame, contours, -1, (0, 255, 0), 1)

    marker = []
    hashMap = hash.HashMap(width_section, height_section)

    # Draw 10 * 10 Section
    for s in range(1, section_num):
        cv.line(frame, (int(width_section * s), 0), (int(width_section * s), height), (0, 255, 255), 1, 1)
        cv.line(frame, (0, int(height_section * s)), (width, int(height_section * s)), (0, 255, 255), 1, 1)

    for i in contours:
        M = cv.moments(i)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            hashMap.insert((cX, cY))
            # cv.circle(frame, (cX, cY), 2, (0, 0, 255), -1)
            # cv.drawContours(frame, [i], 0, (0, 0, 255), 1)

    # Visualize hashed points
    visual_grid = False
    if visual_grid:
        frame_copy = frame.copy()
        for sec in hashMap.grid:
            for point in hashMap.getPointsFromKey(sec):
                color = (0, int(800 * (sec[0] % 2) / section_num), int(800 * (sec[1] % 2) / section_num))
                cv.circle(frame_copy, point, 4, color, -1)
        cv.imshow("grid", frame_copy)

    # Find dot cluster section by section
    for key, points in hashMap.grid.items():
        graph = []
        n = len(points)
        for i in range(n-1):
            for j in range(i+1, n):
                dst = (points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2
                if dst < 200:
                    edge = [points[i], points[j]]
                    cv.line(frame, edge[0], edge[1], (255, 0, 0), 2, 1)
                    graph.append(edge)

    # Show keypoints
    if save_video:
        out.write(frame)
    if visual_video:
        cv.imshow("frame", frame)

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv.destroyAllWindows()
