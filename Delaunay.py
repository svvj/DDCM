# Delaunay

from scipy.spatial import Delaunay
from frame_by_frame import fbf

import numpy as np
import cv2 as cv
import hash
import marker

cap = cv.VideoCapture('data/slow.mp4')
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
out = cv.VideoWriter('output/slow_delaunay.mp4', fourcc, fps, (int(width), int(height)))

save_video = False
visual_video = True
visual_grid = False
marker_visual = False


def visualize_marker(markers):
    dot_size = 4
    circle_type = cv.FILLED
    for node in markers['1']:
        white = (255, 255, 255)
        cv.circle(frame_copy, node, dot_size, white, circle_type)
    for subgraph in markers['2']:
        red = (0, 0, 255)
        cv.circle(frame_copy, subgraph['c'], dot_size, red, circle_type)
    for subgraph in markers['3']:
        green = (0, 255, 0)
        cv.circle(frame_copy, subgraph['c'], dot_size, green, circle_type)
    for subgraph in markers['4']:
        blue = (255, 0, 0)
        cv.circle(frame_copy, subgraph['c'], dot_size, blue, circle_type)


def draw_grid(m, image):
    green = (0, 255, 0)
    row = len(m)
    col = len(m[0])
    for r in range(row):
        for c in range(col):
            if r != row - 1 and m[r][c] != -1 and m[r+1][c] != -1:
                cv.line(image, m[r][c], m[r + 1][c], green, 1)
            if c != col - 1 and m[r][c] != -1 and m[r][c+1] != -1:
                cv.line(image, m[r][c], m[r][c + 1], green, 1)

    cv.imshow("grid", image)


frame_num = 0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame_num += 1
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
            # cv.circle(frame, (cX, cY), 2, (0, 0, 255), -1)
            # cv.drawContours(frame, [i], 0, (0, 0, 255), 1)

    # Visualize hashed points
    if visual_grid:
        for sec in hashMap.grid:
            for point in hashMap.getPointsFromKey(sec):
                color = (0, int(800 * (sec[0] % 2) / section_num), int(800 * (sec[1] % 2) / section_num))
                cv.circle(frame_copy, point, 4, color, -1)
        cv.imshow("grid", frame_copy)

    # Find dot cluster section by section
    visual_marker_edge = True
    for key, points in hashMap.grid.items():
        edges = []
        nodes = []
        n = len(points)
        for i in range(n-1):
            for j in range(i+1, n):
                dst = (points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2
                if dst < 400:
                    nodes.append(points[i])
                    nodes.append(points[j])
                    edge = [points[i], points[j]]
                    if visual_marker_edge:
                        cv.line(frame, edge[0], edge[1], (255, 0, 0), 2, 1)
                    edges.append(edge)

    # Identify markers
    node_sets = set(nodes)
    nodes = list(node_sets)
    v_n = [[1, list(o)] for o in list(set(points) - node_sets)]

    np_nodes = np.array([i[1] for i in v_n])        # numpy nodes array for scipy delaunay triangulation
    subtrees = marker.findSubgraphsInBFS(nodes, edges)
    for i, subtree in enumerate(subtrees):
        if len(subtree['n']) == 2:
            v_n.append([2, subtree['c']])
        elif len(subtree['n']) == 3:
            v_n.append([3, subtree['c']])
        elif len(subtree['n']) == 4:
            v_n.append([4, subtree['c']])
        else:
            continue
        np_nodes = np.concatenate([np_nodes, [np.array(subtree['c'])]], axis=0)

    if frame_num == 1:
        delaunay = Delaunay(np_nodes)
        green = (0, 255, 0)
        np_triangles = np_nodes[delaunay.simplices]
        v_triangles = np.array(v_n, dtype=object)[delaunay.simplices]
        tri_edges = [[[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]] for t in np_triangles]
        v_edges = [[[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]] for t in v_triangles]
        quadrangles = marker.find_quadrangles(v_n, tri_edges, v_edges, frame_copy)
    else:
        if quadrangles:
            quadrangles = fbf(quadrangles)

    # Show keypoints
    if save_video:
        out.write(frame_copy)
    if visual_video:
        draw_grid(quadrangles, frame_copy)

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv.destroyAllWindows()
