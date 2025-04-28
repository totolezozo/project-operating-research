import os
from collections import deque
from typing import List, Optional, Tuple

# --- Vertex labels for n=8 ---
LABELS = ['s','a','b','c','d','e','f','t']
def label(v: int) -> str:
    return LABELS[v]

# --- 1) Read capacity matrix from example1.txt ---
def read_capacity_file(path: str) -> List[List[int]]:
    with open(path, 'r') as f:
        n = int(f.readline().strip())
        C = [list(map(int, f.readline().split())) for _ in range(n)]
    return C

# --- 2) Display helpers ---
def display_matrix(mat: List[List[int]]):
    for row in mat:
        print(" ".join(f"{x:d}" for x in row))

def display_labeled_matrix(mat: List[List[int]]):
    # header
    print("   " + " ".join(f"{label(j):>2s}" for j in range(len(mat))))
    for i, row in enumerate(mat):
        print(f"{label(i):>2s} " + " ".join(f"{x:>2d}" for x in row))


def display_max_flow(C: List[List[int]], R: List[List[int]]):
    """
    C = initial capacity matrix
    R = final residual matrix (after FF completes)
    
    Prints:
      ⋆ Max flow display :
         s   a   b   c   d   e   f   t
       s 0 8/9 5/5 … 
       a 0 0 2/6 …
       …
    """
    n = len(C)
    # 1) Header
    print("⋆ Max flow display :")
    print("   " + " ".join(f"{label(j):>3s}" for j in range(n)))

    # 2) For each row i, compute used=C[i][j]-R[i][j] if C[i][j]>0 else 0
    for i in range(n):
        row_cells = []
        for j in range(n):
            if C[i][j] > 0:
                used = C[i][j] - R[i][j]
                row_cells.append(f"{used}/{C[i][j]}")
            else:
                row_cells.append("0")
        # print row i once
        print(f"{label(i):>2s} " + " ".join(f"{cell:>3s}" for cell in row_cells))

    # 3) Total flow
    total_flow = sum(C[0][j] - R[0][j] for j in range(n))
    print(f"\nValue of the max flow max = {total_flow}\n")


# --- 3) BFS that records parent pointers and prints layers ---
def bfs_with_parents(residual: List[List[int]], s: int, t: int
                    ) -> Tuple[Optional[List[int]], List[int]]:
    n = len(residual)
    parent = [-1]*n
    visited = [False]*n
    q = deque([s])
    visited[s] = True

    print("Breadth-first search :")
    layers = [[s]]
    while q and not visited[t]:
        next_layer = []
        for _ in range(len(q)):
            u = q.popleft()
            for v in range(n):
                if not visited[v] and residual[u][v] > 0:
                    parent[v] = u
                    visited[v] = True
                    q.append(v)
                    next_layer.append(v)
        if next_layer:
            layers.append(next_layer)

    # print each layer
    for layer in layers[1:]:
        names = "".join(label(v) for v in layer)
        parts = "; ".join(f"Π({label(v)}) = {label(parent[v])}" for v in layer)
        print(f"{names} ; {parts}")

    if not visited[t]:
        return None, parent

    # reconstruct path
    path = [t]
    while path[-1] != s:
        path.append(parent[path[-1]])
    path.reverse()
    return path, parent

# --- 4) Ford–Fulkerson with full trace ---
def ford_fulkerson_trace(C: List[List[int]], s: int, t: int
                        ) -> Tuple[int, List[List[int]]]:
    n = len(C)
    R = [row[:] for row in C]       # residual
    max_flow = 0

    print("⋆ Capacity table display :")
    print(n)
    display_matrix(C)
    print("\nThe initial residual graph is the starting graph.\n")

    iteration = 1
    while True:
        print(f"⋆ Iteration {iteration} :")
        path, _ = bfs_with_parents(R, s, t)
        if path is None:
            print("No augmenting path found, terminating.\n")
            break

        # bottleneck Δ
        delta = min(R[path[i]][path[i+1]] for i in range(len(path)-1))
        chain = "".join(label(v) for v in path)
        print(f"Detection of an improving chain : {chain} with a flow {delta}.\n")

        # update residual
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            R[u][v] -= delta
            R[v][u] += delta

        print("Modifications to the residual graph :")
        display_labeled_matrix(R)
        print()
        max_flow += delta
        iteration += 1

    return max_flow, R


def main():
    # locate & read
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_file   = os.path.join(project_dir, "test_files", "example1.txt")

    C = read_capacity_file(data_file)

    # run the trace (this prints only the capacity table, iterations, modifications…)
    max_flow, final_R = ford_fulkerson_trace(C, s=0, t=len(C)-1)

    # now call your new function to print exactly the final matrix
    display_max_flow(C, final_R)


if __name__ == "__main__":
    main()