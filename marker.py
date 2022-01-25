def findSubtreesInBFS(nodes, l_nodes, edges, subNodes, subEdges, subtrees):
    if len(l_nodes) == 0 and len(nodes) == 0:
        return subNodes, subEdges, subtrees
    if len(l_nodes) != 0:
        node = l_nodes[0]
        l_nodes = l_nodes[1:]
    elif len(nodes) != 0 and len(l_nodes) == 0:
        node = nodes[0]
        nodes = nodes[1:]

    if node not in subNodes:
        subNodes.append(node)

    linked_nodes = []
    removable_nodes = []
    removable_edges = []
    for edge in edges:
        if node in edge:
            node_index = edge.index(node)

            if edge not in subEdges:
                subEdges.append(edge)

            if edge[1-node_index] not in subNodes:
                linked_nodes.append(edge[1-node_index])
                removable_nodes.append(edge[1-node_index])
                removable_edges.append(edge)

    if len(removable_nodes) != 0:
        for r_n in removable_nodes:
            if r_n in nodes:
                nodes.remove(r_n)

    if len(removable_edges) != 0:
        for r_e in removable_edges:
            edges.remove(r_e)

    if len(linked_nodes) != 0:
        subNodes, subEdges, subtrees = findSubtreesInBFS(nodes, linked_nodes, edges, subNodes, subEdges, subtrees)

    subtree = {'n': subNodes, 'e': subEdges}
    if len(subtree['e']) != 0:
        subtrees.append(subtree)

    subNodes, subEdges, subtrees = findSubtreesInBFS(nodes, [], edges, [], [], subtrees)

    return subNodes, subEdges, subtrees


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

    _, _, subtrees = findSubtreesInBFS(n_list, [], e_list, [], [], [])

    for st in subtrees:
        e = st['e']
        x = []
        y = []
        for edge in e:
            x.append(edge[0][0])
            x.append(edge[1][0])
            y.append(edge[0][1])
            y.append(edge[1][1])
        plt.plot(x, y)
        plt.show()

