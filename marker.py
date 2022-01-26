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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    e_list = [[(580, 711), (581, 711)], [(598, 665), (593, 655)], [(598, 665), (591, 654)], [(598, 665), (593, 652)], [(593, 655), (591, 654)], [(593, 655), (593, 652)], [(591, 654), (593, 652)], [(479, 643), (487, 640)], [(575, 623), (577, 622)], [(575, 623), (584, 621)], [(577, 622), (584, 621)]]
    n_list = [(487, 640), (581, 711), (591, 654), (433, 652), (575, 623), (479, 643), (593, 655), (593, 652), (498, 710), (595, 713), (577, 622), (598, 665), (584, 621), (456, 652), (580, 711)]
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
    plt.show()
