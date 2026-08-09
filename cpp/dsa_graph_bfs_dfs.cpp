// dsa_graph_bfs_dfs.cpp
// Companion example for: topnotchnote.com/cpp/data_structures_algorithms.html
//
// An adjacency-list graph with both BFS and DFS traversal, so you can see
// the different visiting order each one produces on the same graph.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main dsa_graph_bfs_dfs.cpp

#include <iostream>
#include <list>
#include <queue>
#include <vector>

class Graph {
  int nVertices;
  std::vector<std::list<int>> adj;

  void dfsUtil(int v, std::vector<bool> &visited) const {
    visited[v] = true;
    std::cout << v << " ";
    for (int neighbor : adj[v]) {
      if (!visited[neighbor]) dfsUtil(neighbor, visited);
    }
  }

public:
  explicit Graph(int n) : nVertices(n), adj(n) {}

  void addEdge(int u, int v) {
    adj[u].push_back(v);
    adj[v].push_back(u);   // undirected: add both directions
  }

  void bfs(int start) const {
    std::vector<bool> visited(nVertices, false);
    std::queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
      int v = q.front();
      q.pop();
      std::cout << v << " ";
      for (int neighbor : adj[v]) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          q.push(neighbor);
        }
      }
    }
    std::cout << "\n";
  }

  void dfs(int start) const {
    std::vector<bool> visited(nVertices, false);
    dfsUtil(start, visited);
    std::cout << "\n";
  }
};

int main() {
  //        0
  //      _/ \_
  //     1     2
  //     |     |
  //     3-----4
  Graph g(5);
  g.addEdge(0, 1);
  g.addEdge(0, 2);
  g.addEdge(1, 3);
  g.addEdge(2, 4);
  g.addEdge(3, 4);

  std::cout << "BFS from 0: ";
  g.bfs(0);   // level by level: 0, then its neighbors 1 2, then 3 4

  std::cout << "DFS from 0: ";
  g.dfs(0);   // dives deep first: 0, then 1's branch fully, before backtracking to 2

  return 0;
}
