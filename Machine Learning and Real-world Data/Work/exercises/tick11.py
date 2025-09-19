import os
from typing import Dict, Set
from tick10 import load_graph
from collections import deque

def get_node_betweenness(graph: Dict[int, Set[int]]) -> Dict[int, float]:
    """
    Use Brandes' algorithm to calculate the betweenness centrality for each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to that node's betweenness centrality
    """
    cb = {v: 0 for v in graph}

    for s in graph:
        #Initialization
        S = []
        Q = deque([s])
        dist = {v: -1 for v in graph}
        pred = {v: [] for v in graph}
        sigma = {v: 0 for v in graph}
        delta = {v: 0 for v in graph}
        dist[s] = 0
        sigma[s] = 1

        while Q:
            v = Q.popleft()
            S.append(v)
            for w in graph[v]:
                #Path discovery
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    Q.append(w)
                #Path counting
                if dist[w] == dist[v]+1:
                    sigma[w] = sigma[w] + sigma[v] #Each shortest path to get to us is another shortest path to get to w
                    pred[w].append(v)
        #Accumulation
        while len(S)>0:
            w = S.pop()
            for v in pred[w]:
                delta[v] += (sigma[v]/sigma[w])*(1+delta[w])
            if w != s:
                cb[w] += delta[w]/2
    return cb

def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))
    # graph = load_graph(os.path.join('data', 'social_networks', 'test.edges'))

    betweenness = get_node_betweenness(graph)
    print(f"Node betweenness values: {betweenness}")


if __name__ == '__main__':
    main()