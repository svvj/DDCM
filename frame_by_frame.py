import itertools
from itertools import product

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


def draw_boi(m, margin, w, h, f):
    for r, row in enumerate(m):
        for c, col in enumerate(row):
            lc = max(m[r][c][0] - margin, 0)
            rc = min(m[r][c][0] + margin, w)
            dy = min(m[r][c][1] + margin, h)
            uy = max(m[r][c][1] - margin, 0)
            cv.line(f, (lc, dy), (rc, dy), (0, 0, 255), 1, 1)
            cv.line(f, (lc, uy), (rc, uy), (0, 0, 255), 1, 1)
            cv.line(f, (lc, dy), (lc, uy), (0, 0, 255), 1, 1)
            cv.line(f, (rc, dy), (rc, uy), (0, 0, 255), 1, 1)
    return f


def lost(x, y, m, n_m):
    x_max = len(m[0])
    y_max = len(m)
    if x <= 1 or x >= x_max-1:
        return m[x][y]
    return m[x][y]


def occlusion(m, n_m, vel, num):
    num = num + 1
    if num == 10:
        idx = [(r, row.index(-1)) for r, row in enumerate(n_m) if -1 in row]
        for i in idx:
            n_m[i[0]][i[1]] = m[i[0]][i[1]]
        return n_m

    for r, row in enumerate(n_m):
        for c, node in enumerate(row):
            if node == -1:
                n_list = []
                if r + 2 < len(n_m) and n_m[r + 1][c] != -1 and n_m[r + 2][c] != -1:
                    n_list.append((2 * np.array(n_m[r + 1][c]) - np.array(n_m[r + 2][c])))
                if r - 2 >= 0 and n_m[r - 1][c] != -1 and n_m[r - 2][c] != -1:
                    n_list.append((2 * np.array(n_m[r - 1][c]) - np.array(n_m[r - 2][c])))
                if c + 2 < len(n_m[0]) and n_m[r][c + 1] != -1 and n_m[r][c + 2] != -1:
                    n_list.append((2 * np.array(n_m[r][c + 1]) - np.array(n_m[r][c + 2])))
                if c - 2 >= 0 and n_m[r][c - 1] != -1 and n_m[r][c - 2] != -1:
                    n_list.append((2 * np.array(n_m[r][c - 1]) - np.array(n_m[r][c - 2])))

                if len(n_list) > 0:
                    n_m[r][c] = np.mean(n_list, axis=0, dtype=int).tolist()

    if True in [(-1 in row) for row in n_m]:
        n_m = occlusion(m, n_m, vel, num)

    return n_m



def fbf(nodes, marker, w, h, ca):
    new_marker = (np.ones((8, 12), dtype=int) * -1).tolist()
    vel = np.zeros((8, 12, 2), dtype=int).tolist()
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
                vel[r][c] = (np.array(new_marker[r][c], dtype=int) - np.array(marker[r][c], dtype=int)).tolist()
            # else:
            #     # TODO: Occlusion
            #     # new_marker[r][c] = marker[r][c]
            #     new_marker[r][c] = lost(r, c, marker, new_marker)

            # if type(new_marker[r][c]) == int:
            #     print("asdf")

    if True in [(-1 in row) for row in new_marker]:
        new_marker = occlusion(marker, new_marker, vel, 0)

    cache = draw_boi(new_marker, margin, w, h, cache)

    return cache, new_marker
