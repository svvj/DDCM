# Delaunay

from scipy.spatial import Delaunay
from frame_by_frame import find_dot_cluster, update_quads

import numpy as np
import cv2 as cv
import hash
import marker

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


def identify_marker(i_points, i_nodes, i_edges):
    # Identify markers
    node_sets = set(i_nodes)
    i_nodes = list(node_sets)
    v_n = [[1, list(o)] for o in list(set(i_points) - node_sets)]

    np_nodes = np.array([i[1] for i in v_n])  # numpy nodes array for scipy delaunay triangulation
    subtrees = marker.findSubgraphsInBFS(i_nodes, i_edges)
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
    return v_n, np_nodes


def init_marker(i_points, i_nodes, i_edges):
    v_n, np_nodes = identify_marker(i_points, i_nodes, i_edges)

    delaunay = Delaunay(np_nodes)
    np_triangles = np_nodes[delaunay.simplices]
    v_triangles = np.array(v_n, dtype=object)[delaunay.simplices]
    tri_edges = [[[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]] for t in np_triangles]
    v_edges = [[[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]] for t in v_triangles]
    quadrangles = marker.find_quadrangles(tri_edges, v_edges, frame_copy)
    return quadrangles


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
        quadrangles = init_marker(points, nodes, edges)
    else:
        quadrangles = update_quads(quadrangles)

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
