import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx

""" Graph Representation and Traversal Algorithms """
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.adjMatrix = np.zeros((vertices, vertices), dtype=int)
        self.adjList = {i: [] for i in range(vertices)}
        self.edges = [] 

    def addEdge(self, u, v, directed=False):
        self.adjMatrix[u][v] = 1
        self.adjList[u].append(v)
        self.edges.append((u, v))

        if not directed:
            self.adjMatrix[v][u] = 1
            self.adjList[v].append(u)

    def displayMatrix(self):
        print("Adjacency Matrix:")
        print(self.adjMatrix)
        print("--------------------------------")

    def displayList(self):
        print("Adjacency List:")
        for vertex in range(self.V):
            print(f"Vertex {vertex}: {self.adjList[vertex]}")
        print("--------------------------------")

    def visualize(self):
        """Text-based visualization"""
        print("Graph Structure (Text):")
        for vertex in range(self.V):
            if self.adjList[vertex]:
                print(f"{vertex} -> {' '.join(map(str, self.adjList[vertex]))}")
        print("--------------------------------")

    def drawGraph(self, dfs_path=None, bfs_path=None):
        """Visual graph representation using matplotlib and networkx"""
        plt.figure(figsize=(12, 5))
        
        G = nx.Graph()
        G.add_nodes_from(range(self.V))
        G.add_edges_from(self.edges)
        
        pos = nx.spring_layout(G, seed=42)
        
        plt.subplot(1, 3, 1)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=800, font_size=16, font_weight='bold',
                edge_color='gray', width=2)
        plt.title("Original Graph", fontsize=14, fontweight='bold')
        
        if dfs_path:
            plt.subplot(1, 3, 2)
            node_colors = ['lightgreen' if i in dfs_path else 'lightgray' 
                          for i in range(self.V)]
            nx.draw(G, pos, with_labels=True, node_color=node_colors,
                    node_size=800, font_size=16, font_weight='bold',
                    edge_color='gray', width=2)
            
            dfs_edges = [(dfs_path[i], dfs_path[i+1]) 
                        for i in range(len(dfs_path)-1)
                        if dfs_path[i+1] in self.adjList[dfs_path[i]]]
            nx.draw_networkx_edges(G, pos, dfs_edges, edge_color='red', 
                                  width=3, style='solid')
            
            plt.title(f"DFS Path: {' → '.join(map(str, dfs_path))}", 
                     fontsize=12, fontweight='bold')
        
        if bfs_path:
            plt.subplot(1, 3, 3)
            node_colors = ['lightcoral' if i in bfs_path else 'lightgray' 
                          for i in range(self.V)]
            nx.draw(G, pos, with_labels=True, node_color=node_colors,
                    node_size=800, font_size=16, font_weight='bold',
                    edge_color='gray', width=2)
            
            bfs_edges = [(bfs_path[i], bfs_path[i+1]) 
                        for i in range(len(bfs_path)-1)
                        if bfs_path[i+1] in self.adjList[bfs_path[i]]]
            nx.draw_networkx_edges(G, pos, bfs_edges, edge_color='blue', 
                                  width=3, style='solid')
            
            plt.title(f"BFS Path: {' → '.join(map(str, bfs_path))}", 
                     fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('graph_visualization.png', dpi=300, bbox_inches='tight')
        print("\nGraph visualization saved as 'graph_visualization.png'\n")
        plt.show()


""" Depth-First Search (DFS)"""
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
    print("DFS Iterative Traversal:")

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


""" Breadth-First Search (BFS)"""
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    path = []
    levels = {start: 0}

    visited.add(start)
    print("BFS Traversal:")

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
    print("Graph Representation and Traversal Algorithms")
    print("--------------------------------")

    g = Graph(7)

    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (4, 6), (5, 6)]

    for u, v in edges:
        g.addEdge(u, v)

    print("\nGraph Representation:")
    print("--------------------------------")
    g.displayMatrix()
    print()
    g.displayList()
    print()
    g.visualize()

    print("\n--------------------------------")
    print("Depth-First Search (DFS) - Recursive")
    print("--------------------------------")
    dfsPathRecursive = dfsRecursive(g, 0)
    print(f"\n✓ DFS Recursive Path: {dfsPathRecursive}")

    print("\n--------------------------------")
    print("Depth-First Search (DFS) - Iterative")
    print("--------------------------------")
    dfsPathIterative = dfsIterative(g, 0)
    print(f"\n✓ DFS Iterative Path: {dfsPathIterative}")

    print("\n--------------------------------")
    print("Breadth-First Search (BFS)")
    print("--------------------------------")
    bfsPath, bfsLevels = bfs(g, 0)
    print(f"\n✓ BFS Path: {bfsPath}")
    print(f"✓ BFS Levels: {bfsLevels}")

    # Comparison
    print("\n--------------------------------")
    print("COMPARISON OF RESULTS")
    print("--------------------------------")
    print(f"DFS Recursive Path: {dfsPathRecursive}")
    print(f"DFS Iterative Path: {dfsPathIterative}")
    print(f"BFS Path:           {bfsPath}")
    print("\nKey Observations:")
    print("DFS explores deep into branches before backtracking")
    print("BFS explores level by level (guaranteed shortest path)")
    print("Both visit all reachable vertices but in different orders")

    # Create visualization
    print("\n--------------------------------")
    print("Generating Graph Visualization...")
    print("--------------------------------")
    g.drawGraph(dfs_path=dfsPathRecursive, bfs_path=bfsPath)


if __name__ == "__main__":
    main()
