from collections import deque

import numpy as np

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


def is_appropriate_quad(quad):
    # TODO: Add Condition (i) and Condition (ii) from paper
    vec = quad[:, 0] - quad[:, 1]
    n_v = np.array([v / np.linalg.norm(v) for v in vec])
    in_vec = []
    if quad[0][0] in quad[1]:
        idx0 = int((quad[0][0] == quad[1][1]).all())
        if quad[2][0] in quad[3]:
            in_vec = [quad[0][0] - quad[2][0], quad[0][1] - quad[1][idx0-1]]
        else:
            in_vec = [quad[0][0] - quad[2][1], quad[0][1] - quad[1][idx0-1]]
    else:
        idx0 = int((quad[0][1] == quad[1][1]).all())
        if quad[2][0] in quad[3]:
            in_vec = [quad[0][0] - quad[2][0], quad[0][1] - quad[1][idx0 - 1]]
        else:
            in_vec = [quad[0][0] - quad[2][1], quad[0][1] - quad[1][idx0 - 1]]

    n_in_v = np.array([i_v / np.linalg.norm(i_v) for i_v in in_vec])
    L_Se = 1 - 1/3 * (np.dot(-n_v[0], n_v[1]))**2 - 1/3 * (np.dot(-n_v[2], n_v[3]))**2 - 1/3 * (np.dot(n_in_v[0], n_in_v[1]))**2
    if 0.97 <= L_Se < 1.03:
        return True
    else:
        return False


def find_quadrangles(tri_edges):
    # TODO: Use BFS
    quadrangles = []
    constructed_quad = []
    for i, e in enumerate(tri_edges):
        for j in range(3):
            t_list = list(set(np.where(tri_edges == e[j])[0]))
            quad = np.array([e[(j-1) % 3], e[(j-2) % 3]])
            for t in t_list:
                if i == t or constructed_quad.count([t, i]) != 0:
                    continue
                ad_tri = tri_edges[t]
                for idx, a_t in enumerate(ad_tri):
                    if np.array(a_t == e[j]).all() or (np.array([a_t[1], a_t[0]]) == e[j]).all():
                        ad_idx = idx
                        ad_edges = np.array([ad_tri[(ad_idx-1) % 3], ad_tri[(ad_idx-2) % 3]])
                        quad_edges = np.concatenate((quad, ad_edges), axis=0)
                        constructed_quad.append([i, t])
                        if is_appropriate_quad(quad_edges):
                            quadrangles.append(quad_edges)

    quadrangles = np.array(quadrangles)
    return constructed_quad, quadrangles


def qualify_quadrangles(quad):
    quadrangles = np.array([])
    return quadrangles


if __name__ == "__main__":
    import matplotlib.pyplot as plt

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

    plt.plot(x, y)
    # plt.show()
