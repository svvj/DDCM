# Delaunay
from frame_by_frame import find_dot_cluster, init_marker, update_quads
from marker_array import m_array

import numpy as np
import cv2 as cv
import hash

cap = cv.VideoCapture('240fps.mp4')
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
section_num = 1
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width_section = width / section_num
height_section = height / section_num

fps = cap.get(cv.CAP_PROP_FPS)
print(f"width: {width}, height: {height}, fps: {fps}")
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('delaunay_240.mp4', fourcc, fps, (int(width), int(height)))

save_video = False
visual_video = True
visual_grid = False


def visualize_marker(f, m, n_nodes, pos):
    dot_size = 4
    circle_type = cv.FILLED

    for j, row in enumerate(pos):
        for i, p in enumerate(row):
            for k in range(len(n_nodes)):
                if np.array_equal(n_nodes[k], p):
                    mark = m[k][0]
                    break
            if mark == 1:
                color = (255, 255, 255)     # white
            elif mark == 2:
                color = (0, 0, 255)         # red
            elif mark == 3:
                color = (0, 255, 0)         # green
            else:
                color = (255, 0, 0)         # blue

            cv.circle(f, p, dot_size, color, circle_type)

    cv.imshow("marker", frame_copy)


while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    f_number = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame_copy = frame.copy()

    # Display the resulting frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 110, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    # with_contours = cv.drawContours(frame, contours, -1, (0, 255, 0), 1)

    hashMap = hash.HashMap(width_section, height_section)

    # Draw 10 * 10 Section
    if visual_grid:
        for s in range(1, section_num):
            cv.line(frame_copy, (int(width_section * s), 0), (int(width_section * s), height), (0, 255, 255), 1, 1)
            cv.line(frame_copy, (0, int(height_section * s)), (width, int(height_section * s)), (0, 255, 255), 1, 1)


    for i in contours:
        M = cv.moments(i)
        # Check if it is a closed contour with appropriate area
        if M['m00'] != 0 and cv.contourArea(i) < 100:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            hashMap.insert((cX, cY))

    # Visualize hashed points
    points = []
    for sec in hashMap.grid:
        for point in hashMap.getPointsFromKey(sec):
            points.append(point)
            if visual_grid:
                color = (0, int(800 * (sec[0] % 2) / section_num), int(800 * (sec[1] % 2) / section_num))
                cv.circle(frame_copy, point, 4, color, -1)
    if visual_grid:
        cv.imshow("grid", frame_copy)

    # Find dot cluster section by section
    visual_marker_edge = True
    nodes, edges = find_dot_cluster(hashMap)
    n = len(edges)
    if visual_marker_edge:
        for e in edges:
            cv.line(frame, e[0], e[1], (255, 0, 0), 2, 1)

    green = (0, 255, 0)
    if f_number == 1:
        v_n, n_nodes, m_quads, quadrangles = init_marker(points, nodes, edges, frame_copy)
        unique_markers = np.unique(np.array(quadrangles).reshape(-1, 2), axis=0).reshape(12, 8, 2)
        visualize_marker(frame_copy, v_n, n_nodes, unique_markers)
    else:
        print(quadrangles)
        #quadrangles = update_quads(_, _, quadrangles)

    for q in quadrangles:
        cv.line(frame_copy, q[0][0], q[0][1], green, 1)
        cv.line(frame_copy, q[1][0], q[1][1], green, 1)
        cv.line(frame_copy, q[2][0], q[2][1], green, 1)
        cv.line(frame_copy, q[3][0], q[3][1], green, 1)


    # Show keypoints
    if save_video:
        out.write(frame_copy)
    if visual_video:
        cv.imshow("frame", frame_copy)

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv.destroyAllWindows()
