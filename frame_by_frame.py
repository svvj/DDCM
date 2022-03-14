import numpy as np
import cv2 as cv
import json

with open('m_array.json') as f:
    m_array = json.load(f)['m_array']


fourcc = cv.VideoWriter_fourcc(*'mp4v')


def m_condition(i, n, m_l, m, lc, rc, d, u):
    # if d <= n[0] <= u and lc <= n[1] <= rc and m_l[i] == m:
    if lc <= n[0] <= rc and u <= n[1] <= d:
        return True
    else:
        return False


def draw_marker(f, n_list, m_list):
    dot_size = 4
    circle_type = cv.FILLED
    white = (255, 255, 255)
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    for i, node in enumerate(n_list):
        if m_list[i] == 1:
            cv.circle(f, node, dot_size, white, circle_type)
        if m_list[i] == 2:
            cv.circle(f, node, dot_size, red, circle_type)
        if m_list[i] == 3:
            cv.circle(f, node, dot_size, green, circle_type)
        if m_list[i] == 4:
            cv.circle(f, node, dot_size, blue, circle_type)
    cv.imshow("nodes", f)
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()


def fbf(nodes, marker, w, h, ca):
    new_marker = (np.ones((8, 12), dtype=int) * -1).tolist()
    cache = ca.copy()

    m_list = [p[0] for p in nodes]
    n_list = [p[1] for p in nodes]
    margin = 20

    # draw_marker(ca, n_list, m_list)

    for r, row in enumerate(marker):
        for c, node in enumerate(row):
            lc = max(node[0] - margin, 0)
            rc = min(node[0] + margin, w)
            dy = min(node[1] + margin, h)
            uy = max(node[1] - margin, 0)
            new_node = [n for i, n in enumerate(n_list) if m_condition(i, n, m_list, m_array[r][c], lc, rc, dy, uy)]

            if new_node:
                if len(new_node) == 1:
                    new_marker[r][c] = new_node[0]
                else:
                    np_n = np.unique(np.array(new_node), axis=0)[0]
                    new_marker[r][c] = np_n.tolist()
            else:
                # TODO: Occlusion
                new_marker[r][c] = marker[r][c]

            lc = max(new_marker[r][c][0] - margin, 0)
            rc = min(new_marker[r][c][0] + margin, w)
            dy = min(new_marker[r][c][1] + margin, h)
            uy = max(new_marker[r][c][1] - margin, 0)
            cv.line(cache, (lc, dy), (rc, dy), (0, 0, 255), 1, 1)
            cv.line(cache, (lc, uy), (rc, uy), (0, 0, 255), 1, 1)
            cv.line(cache, (lc, dy), (lc, uy), (0, 0, 255), 1, 1)
            cv.line(cache, (rc, dy), (rc, uy), (0, 0, 255), 1, 1)
    return cache, new_marker
