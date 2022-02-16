from scipy.spatial import Delaunay

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

def update_quads(quad):

    return quad