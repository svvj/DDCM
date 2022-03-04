from queue import Queue
from itertools import combinations

import numpy as np
import cv2 as cv
import json

with open('m_array.json') as f:
    m_array = json.load(f)['m_array']


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


def condition1(e, e_s):
    s = [e[e_s[i]] for i in range(4)]

    if s[0][0] in s[1]:
        A = np.array(s[0][0])
        B = np.array(s[0][1])
        D = np.array(s[1][s[1].index(A.tolist())-1])
    else:
        A = np.array(s[0][1])
        B = np.array(s[0][0])
        D = np.array(s[1][s[1].index(A.tolist()) - 1])
    if s[2][0] in s[3]:
        C = np.array(s[2][0])
    else:
        C = np.array(s[2][1])

    c_1 = False
    AB = (B - A) / np.linalg.norm(B - A)
    CB = (B - C) / np.linalg.norm(B - C)
    AD = (D - A) / np.linalg.norm(D - A)
    CD = (D - C) / np.linalg.norm(D - C)

    if 1-np.square(np.dot(AB, CD)) < 0.01 and 1-np.square(np.dot(AD, CB)) < 0.01:
        c_1 = True
    return c_1


def condition2(e, e_s):
    s = [e[e_s[i]] for i in range(4)]

    if s[0][0] in s[1]:
        A = np.array(s[0][0])
        B = np.array(s[0][1])
        D = np.array(s[1][s[1].index(A.tolist())-1])
    else:
        A = np.array(s[0][1])
        B = np.array(s[0][0])
        D = np.array(s[1][s[1].index(A.tolist()) - 1])
    if s[2][0] in s[3]:
        C = np.array(s[2][0])
    else:
        C = np.array(s[2][1])

    c_2 = False
    AB = (B - A) / np.linalg.norm(B - A)
    CB = (B - C) / np.linalg.norm(B - C)
    AD = (D - A) / np.linalg.norm(D - A)
    CD = (D - C) / np.linalg.norm(D - C)

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
            if [m_array[r][c], m_array[r][c+1], m_array[r+1][c], m_array[r+1][c+1]] == m:
                return [r, c]
    return False


def check_marker(idx, square, s_idx, M):
    r, c = idx
    filled = []
    if M[r][c] != -1:
        filled.append(0)
    if M[r][c+1] != -1:
        filled.append(1)
    if M[r+1][c] != -1:
        filled.append(2)
    if M[r + 1][c + 1] != -1:
        filled.append(3)
    if len(filled) == 0:
        possible_nodes = [0, 1, 2, 3]
        return possible_nodes, M

    possible_edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    comb = set(combinations(filled, 2))
    filled_edges = list(set(possible_edges) & comb)
    possible_nodes = list({0, 1, 2, 3} - set(filled))
    return possible_nodes, M


def set_marker_pos(rc, ed, sq, filled, marker):
    for n_f in filled:
        r, c = rc
        n_s = [ed[sq[i]] for i in range(4)]
        if n_s[0][0] in n_s[1]:
            A = n_s[0][0]
            B = n_s[0][1]
            D = n_s[1][n_s[1].index(A) - 1]
        else:
            A = n_s[0][1]
            B = n_s[0][0]
            D = n_s[1][n_s[1].index(A) - 1]
        if n_s[2][0] in n_s[3]:
            C = n_s[2][0]
        else:
            C = n_s[2][1]

        if n_f == 0:
            marker[r][c] = A
        elif n_f == 1:
            marker[r][c + 1] = D
        elif n_f == 2:
            marker[r + 1][c] = B
        elif n_f == 3:
            marker[r + 1][c + 1] = C
    return marker


def count_not_none(l):
    cnt = 0
    for i in l:
        cnt += sum(x is not None for x in i)
    return cnt


def qualify_quadrangles(input, frame_copy):
    '''
    :param input: markers(index 0) and quadrangles(index 1)
    :param frame_copy: for visual output
    :return: marker info
    '''
    M = (np.ones((8, 12), dtype=int) * -1).tolist()

    m_S = input[:, 0]
    n = m_S.size

    np_S = np.array(input[:, 1].tolist(), dtype=int)

    e = np_S.reshape(-1, 2, 2).tolist()
    S_list = np_S.tolist()
    for edge in e:
        if [edge[1], edge[0]] in e:
            del e[e.index([edge[1], edge[0]])]
    S = change_config_S(e, S_list)
    visited = np.zeros(n)           # check if quadrangle is visited
    m_visited = np.zeros((7, 11), dtype=int).tolist()

    # e : unique edges in list type
    # S : unique quadrangles which are represented by edge indices in list type
    for k in range(n):
        # initialize queue and push one most strict quadrangles
        if visited[k] == 0:
            visited[k] = 1
            cnt = 0

            Q = Queue()
            rc_idx = find_marker(m_S[k])
            if rc_idx and m_visited[rc_idx[0]][rc_idx[1]] == 0:
                filled_marker, M = check_marker(rc_idx, S[k], k, M)
                M = set_marker_pos(rc_idx, e, S[k], filled_marker, M)
                Q.put([rc_idx, S[k], filled_marker])
            else:
                continue

        # clear queue while queue is not empty
        while Q.not_empty:
            rc_idx, s, filled_marker = Q.get()
            m_visited[rc_idx[0]][rc_idx[1]] = 1
            M = set_marker_pos(rc_idx, e, s, filled_marker, M)

            # visualize popped quad from queue on copied frame with color blue

            blue = (255, 0, 0)
            green = (0, 255, 0)
            cache = frame_copy.copy()
            for ed in [s[f] for f in filled_marker]:
                cv.line(frame_copy, e[ed][0], e[ed][1], green, 1)
                cv.line(cache, e[ed][0], e[ed][1], blue, 1)
            cv.imshow("frame1", cache)
            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()

            for i in filled_marker:
                # find adjacent quad which shares edge s[i] and save in variable 'idxs'
                idxs = [index for index, quad in enumerate(S) if s[i] in quad]
                if idxs is not []:
                    for idx in idxs:
                        rc_idx = find_marker(m_S[idx])
                        if rc_idx and m_visited[rc_idx[0]][rc_idx[1]] == 0:
                            S_e = S[idx]

                            # visualize adjacent quad on copied frame with color red
                            red = (0, 0, 255)
                            cache1 = cache.copy()
                            for ed in [S_e[j] for j in range(4)]:
                                cv.line(cache1, e[ed][0], e[ed][1], red, 1)
                            cv.imshow("frame1", cache1)
                            if cv.waitKey(1) == ord('q'):
                                cv.destroyAllWindows()

                            if condition1(e, S_e) or condition2(e, S_e):
                                m_visited[rc_idx[0]][rc_idx[1]] = 1
                                filled_marker, M = check_marker(rc_idx, e, S_e, M)
                                M = set_marker_pos(rc_idx, e, S_e, filled_marker, M)
                                visited[S.index(S_e)] = 1
                                Q.put([rc_idx, S_e, filled_marker])

                                for ed in [S_e[j] for j in range(4)]:
                                    cv.line(frame_copy, e[ed][0], e[ed][1], green, 1)
                                cv.imshow("frame1", frame_copy)
                                if cv.waitKey(1) == ord('q'):
                                    cv.destroyAllWindows()
                            break
                        else:
                            continue
                print(cnt)
                cnt += 1
            if Q.empty():
                print('queue is empty')
                break

    cv.destroyAllWindows()
    for row in M:
        for col in row:
            cv.circle(frame_copy, col, 4, red, -1)
    cv.imshow("frame2", frame_copy)
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()

    print(M)
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