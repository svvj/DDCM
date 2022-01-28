# Delaunay

from scipy.spatial import Delaunay

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


while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
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
                if dst < 300:
                    nodes.append(points[i])
                    nodes.append(points[j])
                    edge = [points[i], points[j]]
                    if visual_marker_edge:
                        cv.line(frame, edge[0], edge[1], (255, 0, 0), 2, 1)
                    edges.append(edge)

        # Identify markers
        markers = {'1': [], '2': [], '3': [], '4': []}
        node_sets = set(nodes)      # Remove duplicates
        nodes = list(node_sets)
        markers['1'] = list(set(points) - node_sets)        # Saving single points which are not constructing edges
        np_nodes = np.array([list(i) for i in markers['1']])

        subtrees = marker.findSubgraphsInBFS(nodes, edges)
        for i, subtree in enumerate(subtrees):
            if len(subtree['n']) == 2:
                np_nodes = np.concatenate([np_nodes, [np.array(subtree['c'])]], axis=0)
                markers['2'].append(subtree)
            elif len(subtree['n']) == 3:
                np_nodes = np.concatenate([np_nodes, [np.array(subtree['c'])]], axis=0)
                markers['3'].append(subtree)
            elif len(subtree['n']) == 4:
                np_nodes = np.concatenate([np_nodes, [np.array(subtree['c'])]], axis=0)
                markers['4'].append(subtree)
            else:
                continue

        if marker_visual:
            visualize_marker(markers)

        delaunay = Delaunay(np_nodes)
        green = (0, 255, 0)
        triangles = np_nodes[delaunay.simplices]
        tri_edges = np.array([[[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]] for t in triangles])
        # for t in tri_edges:
        #     cv.line(frame_copy, t[0][0], t[0][1], green, 1)
        #     cv.line(frame_copy, t[1][0], t[1][1], green, 1)
        #     cv.line(frame_copy, t[2][0], t[2][1], green, 1)
        con_quads, quadrangles = marker.find_quadrangles(tri_edges)

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
