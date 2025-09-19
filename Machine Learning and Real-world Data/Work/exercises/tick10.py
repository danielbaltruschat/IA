import os
from typing import Dict, Set


def load_graph(filename: str) -> Dict[int, Set[int]]:
    """
    Load the graph file. Each line in the file corresponds to an edge; the first column is the source node and the
    second column is the target. As the graph is undirected, your program should add the source as a neighbour of the
    target as well as the target a neighbour of the source.

    @param filename: The path to the network specification
    @return: a dictionary mapping each node (represented by an integer ID) to a set containing all the nodes it is
        connected to (also represented by integer IDs)
    """
    output = {}
    with open(filename, encoding='utf-8') as f:
        content = f.readlines()
        for s in content:
            parts = s.split(" ")
            u = int(parts[0])
            v = int(parts[1])

            if u not in output:
                output[u] = set()
            output[u].add(v)
            if v not in output:
                output[v] = set()
            output[v].add(u)
    return output


def get_node_degrees(graph: Dict[int, Set[int]]) -> Dict[int, int]:
    """
    Find the number of neighbours of each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to the degree of the node
    """
    return dict((k, len(v)) for k, v in graph.items())


def get_diameter(graph: Dict[int, Set[int]]) -> int:
    """
    Find the longest shortest path between any two nodes in the network using a breadth-first search.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the length of the longest shortest path between any pair of nodes in the graph
    """
    m = 0
    for n in graph:
        m = max(m, bfs_diameter(graph, n))
    return m



def bfs_diameter(graph: Dict[int, Set[int]], s: int) -> int:
    q = [s]
    visited = {s}
    depth = -1

    while q:
        for _ in range(len(q)):
            current = q.pop()
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.insert(0, neighbor)
        depth += 1

    return depth


def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))

    degrees = get_node_degrees(graph)
    print(f"Node degrees: {degrees}")

    diameter = get_diameter(graph)
    print(f"Diameter: {diameter}")


if __name__ == '__main__':
    main()