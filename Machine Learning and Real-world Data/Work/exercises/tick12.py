import os
from collections import deque
from typing import Set, Dict, List, Tuple
from tick10 import load_graph


def get_number_of_edges(graph: Dict[int, Set[int]]) -> int:
    """
    Find the number of edges in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the number of edges
    """
    total = 0
    for v in graph:
        total += len(graph[v])
    return total//2


def get_components(g: Dict[int, Set[int]]) -> List[Set[int]]:
    """
    Find the number of components in the graph using a DFS.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: list of components for the graph.
    """
    out = []
    graph = g.copy()
    while graph:
        s = list(graph.keys())[0]
        stack = [s]
        nodesInThisTree = set()
        while stack:
            current = stack.pop()
            if current not in graph:
                continue
            for n in graph[current]:
                if n in graph:
                    stack.append(n)
            del graph[current]
            nodesInThisTree.add(current)
        out.append(nodesInThisTree)
    return out



def get_edge_betweenness(graph: Dict[int, Set[int]]) -> Dict[Tuple[int, int], float]:
    """
    Calculate the edge betweenness.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: betweenness for each pair of vertices in the graph connected to each other by an edge
    """
    cb = {}

    for s in graph: #V
        #Initialization
        S = []
        Q = deque([s])
        dist = {v: -1 for v in graph}
        pred = {v: [] for v in graph}
        sigma = {v: 0 for v in graph}
        delta = {v: 0 for v in graph}
        dist[s] = 0
        sigma[s] = 1

        while Q: #v
            v = Q.popleft()
            S.append(v)
            for w in graph[v]:
                #Path discovery
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    Q.append(w)
                #Path counting
                if dist[w] == dist[v]+1:
                    sigma[w] = sigma[w] + sigma[v]
                    pred[w].append(v)
        #Accumulation
        while len(S)>0:
            w = S.pop()
            for v in pred[w]:
                c = (sigma[v]/sigma[w])*(1+delta[w])
                if (v,w) not in cb:
                    cb[(v,w)]=0
                cb[(v,w)] += c
                delta[v] += c
    return cb


def girvan_newman(graph: Dict[int, Set[int]], min_components: int) -> List[Set[int]]:
    """     * Find the number of edges in the graph.
     *
     * @param graph
     *        {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *        loaded graph
     * @return {@link Integer}> Number of edges.
    """
    while len(get_components(graph)) < min_components:
        eb = get_edge_betweenness(graph)
        k = max(eb,key=eb.get)
        threshold = eb[k]-1e-10
        toRemove = []
        for k in eb:
            if eb[k]>threshold:
                toRemove.append(k)
        for (u,v) in toRemove:
            graph[u].remove(v)

    return get_components(graph)


def main():
    # graph = load_graph(os.path.join('data', 'social_networks', 'facebook_circle.edges'))
    graph = load_graph(os.path.join('data', 'social_networks', 'test.edges'))

    num_edges = get_number_of_edges(graph)
    print(f"Number of edges: {num_edges}")

    components = get_components(graph)
    print(f"Number of components: {len(components)}")

    edge_betweenness = get_edge_betweenness(graph)
    print(f"Edge betweenness: {edge_betweenness}")

    clusters = girvan_newman(graph, min_components=2)
    print(f"Girvan-Newman for 20 clusters: {clusters}")


if __name__ == '__main__':
    main()