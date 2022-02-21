from queue import Queue

import numpy as np
import cv2 as cv

def mean_int(list):
    return int(sum(list) / len(list))

def findSubgraphsInBFS(nodes, edges):
    i = 0
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
    if quad[0][0] in quad[1]:
        idx0 = int((quad[0][0] == quad[1][1]).all())
        a = [0, 0]
        b = [0, 1]
        d = [1, idx0-1]
        if quad[2][0] in quad[3]:
            c = [2, 0]
        else:
            c = [2, 1]
    else:
        idx0 = int((quad[0][1] == quad[1][1]).all())
        a = [0, 1]
        b = [0, 0]
        d = [1, idx0 - 1]
        if quad[2][0] in quad[3]:
            c = [2, 0]
        else:
            c = [2, 1]
    A = quad[a[0]][a[1]]
    B = quad[b[0]][b[1]]
    C = quad[c[0]][c[1]]
    D = quad[d[0]][d[1]]

    in_vec = [C - A, B - D]
    marker_id = np.array([m[a[0]][a[1]], m[b[0]][b[1]], m[c[0]][c[1]], m[d[0]][d[1]]])

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

def find_e_hat(s_e, s, l):
    s_l_mid = s[l][1] + (s[l][0] - s[l][1]) / 2
    s_mid = (s[0][0] + s[2][0]) / 2
    s_e_mid = np.array([(u[0] + u[1])/2 for u in s_e])
    hat_mid = np.array(s_l_mid + (s_l_mid - s_mid))
    for i in range(4):
        if np.linalg.norm(hat_mid - s_e_mid[i]) < 2:
            return i
    return False


def verify_quadrangles(quads):
    # TODO: verify quads and give ids
    return quads


def qualify_quadrangles(S, frame_copy):
    m_S = S[:, 0]
    S = S[:, 1]
    n = S.size
    e = np.array([[q[0][1], q[1][1]] for q in S])
    visited = np.zeros(n)
    M_S = []
    M = []
    for k in range(S.size):
        if len(M) == 77:
            break
        if visited[k] == 0:
            M.append(verify_quadrangles(S[k]))
            M_S.append(m_S[k])
            Q = Queue()
            Q.put(S[k])
            visited[k] = 1
            cnt = 0
            while Q.not_empty:
                s = Q.get()
                green = (0, 255, 0)
                for q in s:
                    for i in range(4):
                        cv.line(frame_copy, q[0], q[1], green, 1)
                cv.imshow("frame1", frame_copy)
                if cv.waitKey(1) == ord('q'):
                    cv.destroyAllWindows()
                for i in range(4):
                    idx = find_index(e, s[i])
                    if idx:
                        S_e = S[idx]
                        if condition1(S_e) or condition2(S_e):
                            # TODO: give id in verified quads during BFS
                            e_hat = S_e[find_e_hat(S_e, s, i)]
                            h_idx = find_index(e, e_hat)
                            if h_idx:
                                if visited[h_idx] == 0 and condition1(S[h_idx]):
                                    M.append(verify_quadrangles(S[h_idx]))
                                    M_S.append(m_S[h_idx])
                                    Q.put(S[h_idx])
                                    visited[h_idx] = 1

                print(cnt)
                cnt += 1
                if Q.empty():
                    print('queue is empty')
                    break
    return M, M_S


def find_quadrangles(tri_edges, v_edges, frame_copy):
    # TODO: Use BFS
    quadrangles = []
    constructed_quad = []
    for i, e in enumerate(tri_edges):
        for j in range(3):
            t_list = list(set(np.where(np.array(tri_edges) == np.array(e[j]))[0]))
            quad = np.array([v_edges[i][(j - 1) % 3], v_edges[i][(j - 2) % 3]], dtype=object)
            for t in t_list:
                if i == t or constructed_quad.count([t, i]) != 0:
                    continue
                ad_tri = tri_edges[t]
                for idx, a_t in enumerate(ad_tri):
                    if (np.array(a_t) == e[j]).all() or (np.array([a_t[1], a_t[0]]) == e[j]).all():
                        ad_idx = idx
                        ad_edges = np.array([v_edges[t][(ad_idx - 1) % 3], v_edges[t][(ad_idx - 2) % 3]], dtype=object)
                        quad_edges = np.concatenate((quad, ad_edges), axis=0)
                        constructed_quad.append([i, t])
                        a_quad, L, seq_quad, m_id = is_appropriate_quad(quad_edges)
                        if a_quad:
                            quadrangles.append([L, m_id, seq_quad])

    quadrangles = np.array(quadrangles, dtype=object)
    sorted_quads = quadrangles[quadrangles[:, 0].argsort()][::-1][:, 1:3]
    q_quads, m_quads = qualify_quadrangles(sorted_quads, frame_copy)
    return q_quads, m_quads


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