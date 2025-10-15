import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.adjMatrix = np.zeros((vertices, vertices), dtype=int)
        self.adjList = {i: [] for i in range(vertices)}
    
    def addEdge(self, u, v, directed=False):
        self.adjMatrix[u][v] = 1
        self.adjList[u].append(v)
        
        if not directed:
            self.adjMatrix[v][u] = 1
            self.adjList[v].append(u)
    
    def displayMatrix(self):
        print("Adjacency Matrix:")
        print(self.adjMatrix)
    
    def displayList(self):
        print("\nAdjacency List:")
        for vertex in range(self.V):
            print(f"Vertex {vertex}: {self.adjList[vertex]}")
    
    def visualize(self):
        print("\nGraph Structure:")
        for vertex in range(self.V):
            if self.adjList[vertex]:
                print(f"{vertex} -> {' '.join(map(str, self.adjList[vertex]))}")

def dfsRecursive(graph, start, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []
    
    visited.add(start)
    path.append(start)
    print(f"Visiting vertex: {start}")
    
    for neighbor in graph.adjList[start]:
        if neighbor not in visited:
            dfsRecursive(graph, neighbor, visited, path)
    
    return path


def dfsIterative(graph, start):
    visited = set()
    stack = [start]
    path = []
    
    print("\nDFS Iterative Traversal:")
    
    while stack:
        vertex = stack.pop()
        
        if vertex not in visited:
            visited.add(vertex)
            path.append(vertex)
            print(f"Visiting vertex: {vertex}, Stack: {stack}")
            
            for neighbor in reversed(graph.adjList[vertex]):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return path

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    path = []
    levels = {start: 0} 
    
    visited.add(start)
    print("\nBFS Traversal:")
    
    while queue:
        vertex = queue.popleft()
        path.append(vertex)
        print(f"Visiting vertex: {vertex}, Queue: {list(queue)}, Level: {levels[vertex]}")
        
        for neighbor in graph.adjList[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                levels[neighbor] = levels[vertex] + 1
    
    return path, levels


def main():
    
    print("--------------------------------")
    print("SEARCH ALGORITHMS DEMONSTRATION")
    print("--------------------------------")
    
    g = Graph(7)
    
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (2, 5), (4, 6), (5, 6)]
    
    for u, v in edges:
        g.addEdge(u, v)
    
    print("\n--- Graph Information ---")
    g.displayMatrix()
    g.displayList()
    g.visualize()
    
    # Test DFS Recursive
    print("\n--------------------------------")
    print("DEPTH-FIRST SEARCH (RECURSIVE)")
    print("--------------------------------")
    dfs_path_recursive = dfsRecursive(g, 0)
    print(f"DFS Path (Recursive): {dfs_path_recursive}")
    
    # Test DFS Iterative
    print("\n--------------------------------")
    print("DEPTH-FIRST SEARCH (ITERATIVE)")
    print("--------------------------------")
    dfs_path_iterative = dfsIterative(g, 0)
    print(f"DFS Path (Iterative): {dfs_path_iterative}")
    
    # Test BFS
    print("\n--------------------------------")
    print("BREADTH-FIRST SEARCH")
    print("--------------------------------")
    bfs_path, levels = bfs(g, 0)
    print(f"BFS Path: {bfs_path}")
    print(f"Levels from start: {levels}")
    
    # Compare results
    print("\n--------------------------------")
    print("COMPARISON")
    print("--------------------------------")
    print(f"DFS Path: {dfs_path_recursive}")
    print(f"BFS Path: {bfs_path}")


if __name__ == "__main__":
    main()