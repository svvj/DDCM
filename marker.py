def findSubtreesInBFS(nodes, edges):
    i = 0
    subtrees = []

    for node in nodes:
        linked_nodes = []
        subtree = {'n': [], 'e': []}

        for edge in edges:
            if node in edge:
                node_index = edge.index(node)

                if node not in subtree['n']:
                    subtree['n'].append(node)
                if edge not in subtree['e']:
                    subtree['e'].append(edge)

                if edge[1-node_index] not in subtree['n']:
                    subtree['n'].append(edge[1-node_index])
                    linked_nodes.append(edge[1-node_index])

        # findSubtreesInBFS(linked_nodes, edges)

        if len(subtree['n']) != 0:
            subtrees.append(subtree)

    return subtrees


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    e_list = [[(580, 711), (581, 711)], [(598, 665), (593, 655)], [(598, 665), (591, 654)], [(598, 665), (593, 652)], [(593, 655), (591, 654)], [(593, 655), (593, 652)], [(591, 654), (593, 652)], [(479, 643), (487, 640)], [(575, 623), (577, 622)], [(575, 623), (584, 621)], [(577, 622), (584, 621)]]
    n_list = [(487, 640), (581, 711), (591, 654), (433, 652), (575, 623), (479, 643), (593, 655), (593, 652), (498, 710), (595, 713), (577, 622), (598, 665), (584, 621), (456, 652), (580, 711)]
    subtrees = findSubtreesInBFS(n_list, e_list)

    x = []
    y = []
    for subtree in subtrees:
        e = subtree['e']
        for edge in e:
            x.append(edge[0][0])
            x.append(edge[1][0])
            y.append(edge[0][1])
            y.append(edge[1][1])

    plt.plot(x, y)
    plt.show()
