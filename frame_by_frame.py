from scipy.spatial import Delaunay
from marker import find_quadrangles, findSubgraphsInBFS

import numpy as np
import cv2 as cv
import hash


def find_dot_cluster(hashmap):
    for key, points in hashmap.grid.items():
        nodes = []
        edges = []

        n = len(points)
        for i in range(n-1):
            for j in range(i+1, n):
                dst = (points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2
                if dst < 300:
                    nodes.append(points[i])
                    nodes.append(points[j])
                    edge = [points[i], points[j]]
                    edges.append(edge)
    return nodes, edges


def identify_marker(i_points, i_nodes, i_edges):
    # Identify markers
    node_sets = set(i_nodes)
    i_nodes = list(node_sets)
    v_n = [[1, list(o)] for o in list(set(i_points) - node_sets)]

    np_nodes = np.array([i[1] for i in v_n])  # numpy nodes array for scipy delaunay triangulation
    subtrees = findSubgraphsInBFS(i_nodes, i_edges)
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


def init_marker(i_points, i_nodes, i_edges, frame_copy):
    v_n, np_nodes = identify_marker(i_points, i_nodes, i_edges)

    delaunay = Delaunay(np_nodes)
    np_triangles = np_nodes[delaunay.simplices]
    v_triangles = np.array(v_n, dtype=object)[delaunay.simplices]
    tri_edges = [[[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]] for t in np_triangles]
    v_edges = [[[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]] for t in v_triangles]
    quadrangles = find_quadrangles(tri_edges, v_edges, frame_copy)
    return v_n, np_nodes, quadrangles


def update_quads(n, np_n, quad):
    # print(n)
    return quad