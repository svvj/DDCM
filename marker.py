from queue import Queue
from itertools import combinations

import numpy as np
import cv2 as cv
import json

with open('m_array.json') as f:
    m_array = json.load(f)['m_array']

M = (np.ones((8, 12), dtype=int) * -1).tolist()

def mean_int(list):
    return int(sum(list) / len(list))

def findSubgraphsInBFS(nodes, edges):
    subgraphs = []

    for node in nodes:
        linked_nodes = []
        subgraph = {'n': [], 'e': [], 'c': ()}

        for edge in edges:
            if node in edge:
                node_index = edge.index(node)

                if node not in subgraph['n']:
                    subgraph['n'].append(node)
                if edge not in subgraph['e']:
                    subgraph['e'].append(edge)

                if edge[1-node_index] not in subgraph['n']:
                    subgraph['n'].append(edge[1-node_index])
                    linked_nodes.append(edge[1-node_index])

        if len(subgraph['n']) != 0:
            subgraph['c'] = list(map(mean_int, zip(*subgraph['n'])))
            subgraphs.append(subgraph)

    return subgraphs


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def is_appropriate_quad(v_quad):
    # TODO: make unique edge list and return
    m = np.array(v_quad[:, :, 0].tolist())
    quad = np.array(v_quad[:, :, 1].tolist())
    vec = quad[:, 0] - quad[:, 1]
    n_v = np.array([v / np.linalg.norm(v) for v in vec])

    nodes = np.unique(quad.reshape(-1).reshape(-1, 2), axis=0)
    A = nodes[0] if (nodes[0][1] <= nodes[1][1]) else nodes[1]
    a = [quad.reshape(-1, 2).tolist().index(A.tolist()) // 2, quad.reshape(-1, 2).tolist().index(A.tolist()) % 2]
    A_near = [i for i, value in enumerate(quad.reshape(-1, 2).tolist()) if value == A.tolist()]

    prob_bd = [[A_near[0] // 2, 1-(A_near[0] % 2)], [A_near[1] // 2, 1-(A_near[1] % 2)]]
    prob_BD = [quad[prob_bd[0][0]][prob_bd[0][1]], quad[prob_bd[1][0]][prob_bd[1][1]]]

    B, D, b, d = (prob_BD[0], prob_BD[1], prob_bd[0], prob_bd[1]) if prob_BD[0][1] <= prob_BD[1][1] \
                 else (prob_BD[1], prob_BD[0], prob_bd[1], prob_bd[0])
    C = np.array([c for c in nodes if c.tolist() not in [A.tolist(), B.tolist(), D.tolist()]]).reshape(2)
    c = [quad.reshape(-1, 2).tolist().index(C.tolist()) // 2, quad.reshape(-1, 2).tolist().index(C.tolist()) % 2]

    in_vec = [C - A, B - D]

    marker_id = [m[a[0]][a[1]], m[b[0]][b[1]], m[d[0]][d[1]], m[c[0]][c[1]]]

    n_in_v = np.array([i_v / np.linalg.norm(i_v) for i_v in in_vec])
    L_Se = 1 - 1 / 3 * (np.dot(-n_v[0], n_v[1])) ** 2 \
           - 1 / 3 * (np.dot(-n_v[2], n_v[3])) ** 2 \
           - 1 / 3 * (np.dot(n_in_v[0], n_in_v[1])) ** 2
    if 0.6 <= L_Se:
        return True, L_Se, np.array([[A, B], [A, D], [C, B], [C, D]]), marker_id
    else:
        return False, L_Se, np.array([[A, B], [A, D], [C, B], [C, D]]), marker_id


def find_index(e, s):
    for i, n in enumerate(e):
        if np.equal(n, s).all() or np.equal(np.array([n[1], n[0]]), s).all():
            return i
    return False


def condition1(s):
    c_1 = False
    AB = (s[0][1] - s[0][0]) / np.linalg.norm(s[0][1] - s[0][0])
    CB = (s[0][1] - s[2][0]) / np.linalg.norm(s[0][1] - s[2][0])
    AD = (s[1][1] - s[0][0]) / np.linalg.norm(s[1][1] - s[0][0])
    CD = (s[1][1] - s[2][0]) / np.linalg.norm(s[1][1] - s[2][0])

    if 1-np.square(np.dot(AB, CD)) < 0.01 and 1-np.square(np.dot(AD, CB)) < 0.01:
        c_1 = True
    return c_1


def condition2(s):
    c_2 = False
    AB = (s[0][1] - s[0][0]) / np.linalg.norm(s[0][1] - s[0][0])
    CB = (s[0][1] - s[2][0]) / np.linalg.norm(s[0][1] - s[2][0])
    AD = (s[1][1] - s[0][0]) / np.linalg.norm(s[1][1] - s[0][0])
    CD = (s[1][1] - s[2][0]) / np.linalg.norm(s[1][1] - s[2][0])

    if np.square(np.dot(AD, CD)) < 0.01 and 1-np.square(np.dot(AB, CB)) < 0.01:
        c_2 = True
    return c_2


def change_config_S(edges, quads):
    new_quads = []
    for q in quads:
        new_q = []
        for eg in q:
            if [eg[1], eg[0]] in edges:
                new_q.append(edges.index([eg[1], eg[0]]))
            elif eg in edges:
                new_q.append(edges.index(eg))
        new_quads.append(new_q)
    return new_quads


def find_e_hat(s_e, s, l):
    s_l_mid = s[l][1] + (s[l][0] - s[l][1]) / 2
    s_mid = (s[0][0] + s[2][0]) / 2
    s_e_mid = np.array([(u[0] + u[1])/2 for u in s_e])
    hat_mid = np.array(s_l_mid + (s_l_mid - s_mid))
    for i in range(4):
        if np.linalg.norm(hat_mid - s_e_mid[i]) < 2:
            return i
    return False


def find_marker(m):
    row, col = 7, 11
    for r in range(row):
        for c in range(col):
            if np.array_equal(np.array([m_array[r][c], m_array[r][c+1], m_array[r+1][c], m_array[r+1][c+1]]), m):
                return [r, c]
    return False


def check_marker(idx, square, s_idx):
    global M

    r, c = idx
    filled = []
    if M[r][c] != -1:
        filled.append(0)
    if M[r + 1][c] != -1:
        filled.append(1)
    if M[r][c + 1] != -1:
        filled.append(2)
    if M[r + 1][c + 1] != -1:
        filled.append(3)
    if len(filled) == 0:
        possible_edges = [0, 1, 2, 3]
        return possible_edges, False

    M[r][c] = square[s_idx][0][0]           # A
    M[r + 1][c] = square[s_idx][0][1]       # B
    M[r][c + 1] = square[s_idx][1][1]       # D
    M[r + 1][c + 1] = square[s_idx][2][0]   # C

    not_filled = list(set([0, 1, 2, 3]) - set(filled))
    possible_edges = {(0, 1), (0, 2), (1, 3), (2, 3)}
    comb = set(combinations(filled, 2))
    filled_edges = list(possible_edges & comb)
    return filled_edges, True


def count_not_none(l):
    cnt = 0
    for i in l:
        cnt += sum(x is not None for x in i)
    return cnt


def qualify_quadrangles(input, frame_copy):
    global M

    m_S = input[:, 0]
    n = m_S.size

    np_S = np.array(input[:, 1].tolist(), dtype=int)

    e_list = np_S.reshape(-1, 2, 2).tolist()
    S_list = np_S.tolist()
    for edge in e_list:
        if [edge[1], edge[0]] in e_list:
            del e_list[e_list.index([edge[1], edge[0]])]
    S = change_config_S(e_list, S_list)
    visited = np.zeros(n)
    m_visited = np.zeros((7, 11), dtype=int).tolist()
    for k in range(n):
        if visited[k] == 0:
            Q = Queue()
            rc_idx = find_marker(m_S[k])
            if rc_idx:
                filled_marker, is_filled = check_marker(rc_idx, S[k], k)
                Q.put([rc_idx, S[k], filled_marker])
            else:
                continue

            visited[k] = 1
            cnt = 0
            while Q.not_empty:
                rc_idx, s, filled_marker = Q.get()
                m_visited[rc_idx[0]][rc_idx[1]] = 1

                green = (0, 255, 0)
                for q in [s[f] for f in filled_marker]:
                    cv.line(frame_copy, q[0], q[1], green, 1)
                cv.imshow("frame1", frame_copy)
                if cv.waitKey(1) == ord('q'):
                    cv.destroyAllWindows()

                for i in filled_marker:
                    idx = find_index(e, s[i])
                    if idx:
                        S_e = S[idx]

                        red = (0, 0, 255)
                        for q in [s[f] for f in filled_marker]:
                            cv.line(frame_copy, q[0], q[1], red, 1)
                        cv.imshow("frame1", frame_copy)
                        if cv.waitKey(1) == ord('q'):
                            cv.destroyAllWindows()

                        if condition1(S_e) or condition2(S_e):
                            e_hat = S_e[find_e_hat(S_e, s, i)]
                            h_idx = find_index(e, e_hat)
                            if h_idx:
                                if visited[h_idx] == 0 and condition1(S[h_idx]):
                                    rc_idx = find_marker(m_S[h_idx])

                                    green = (0, 255, 0)
                                    for q in [s[f] for f in filled_marker]:
                                        cv.line(frame_copy, q[0], q[1], green, 1)
                                    cv.imshow("frame1", frame_copy)
                                    if cv.waitKey(1) == ord('q'):
                                        cv.destroyAllWindows()

                                    if rc_idx:
                                        filled_marker, is_filled = check_marker(rc_idx, S[h_idx], h_idx)
                                    else:
                                        continue
                                    Q.put([rc_idx, S[k], filled_marker])
                                    visited[h_idx] = 1
                                    m_visited[rc_idx[0]][rc_idx[1]] = 1

                print(cnt)
                cnt += 1
                if Q.empty():
                    print('queue is empty')
                    break
    return M


def find_quadrangles(tri_edges, v_edges, frame_copy, markers):
    # TODO: Use BFS
    quadrangles = []
    constructed_quad = []
    for i, e in enumerate(tri_edges):
        for j in range(3):
            t_list = list(set(np.where(np.array(tri_edges) == np.array(e[j]))[0]))
            quad = np.array([v_edges[i][(j-1) % 3], v_edges[i][(j-2) % 3]], dtype=object)
            for t in t_list:
                if i == t or constructed_quad.count([t, i]) != 0:
                    continue
                ad_tri = tri_edges[t]
                for idx, a_t in enumerate(ad_tri):
                    if (np.array(a_t) == e[j]).all() or (np.array([a_t[1], a_t[0]]) == e[j]).all():
                        ad_idx = idx
                        ad_edges = np.array([v_edges[t][(ad_idx-1) % 3], v_edges[t][(ad_idx-2) % 3]], dtype=object)
                        quad_edges = np.concatenate((quad, ad_edges), axis=0)
                        constructed_quad.append([i, t])
                        a_quad, L, seq_quad, m_id = is_appropriate_quad(quad_edges)
                        if a_quad:
                            quadrangles.append([L, m_id, seq_quad])

    quadrangles = np.array(quadrangles, dtype=object)
    sorted_quads = quadrangles[quadrangles[:, 0].argsort()][::-1][:, 1:3]
    q_quads = qualify_quadrangles(sorted_quads, frame_copy)
    return q_quads

if __name__ == "__main__":
    n_list = [(0, 0), (0, 1), (0, 2), (0, 3),
              (1, 0), (1, 1), (1, 2), (1, 3),
              (2, 0), (2, 1), (2, 2), (2, 3),
              (3, 0), (3, 1), (3, 2), (3, 3)]
    e_list = [[(0, 0), (0, 1)], [(0, 0), (1, 0)], [(1, 0), (1, 1)], [(0, 1), (1, 1)],
              [(0, 2), (0, 3)], [(0, 2), (1, 2)], [(1, 2), (1, 3)], [(1, 3), (0, 3)],
              [(2, 0), (2, 1)], [(2, 0), (3, 0)], [(3, 0), (3, 1)], [(3, 1), (2, 1)],
              [(2, 2), (2, 3)], [(2, 2), (3, 2)], [(3, 2), (3, 3)], [(3, 3), (2, 3)]]
    subgraphs = findSubgraphsInBFS(n_list, e_list)

    x = []
    y = []
    for subgraph in subgraphs:
        e = subgraph['e']
        for edge in e:
            x.append(edge[0][0])
            x.append(edge[1][0])
            y.append(edge[0][1])
            y.append(edge[1][1])