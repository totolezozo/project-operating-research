import os
from collections import deque
from typing import List, Optional, Tuple
import math

# --- Vertex labels for n up to 26 (s, a, b, ..., z) ---
LABELS = ['s'] + [chr(ord('a') + i) for i in range(25)]
def label(v: int) -> str:
    return LABELS[v]

# --- 1) Read capacity-only file ---
def read_capacity_file(path: str) -> List[List[int]]:
    with open(path, 'r') as f:
        n = int(f.readline().strip())
        C = [list(map(int, f.readline().split())) for _ in range(n)]
    return C

# --- 1b) Read min-cost instance (C then D) ---
def read_min_cost_file(path: str) -> Tuple[int, List[List[int]], List[List[int]]]:
    with open(path, 'r') as f:
        n1 = int(f.readline().strip())
        C = [list(map(int, f.readline().split())) for _ in range(n1)]
        n2 = int(f.readline().strip()); assert n1 == n2
        D = [list(map(int, f.readline().split())) for _ in range(n1)]
    return n1, C, D

# --- 2) Display helpers ---
def display_matrix(mat: List[List[int]]):
    for row in mat:
        print(" ".join(f"{x:>4d}" for x in row))

def display_labeled_matrix(mat: List[List[int]]):
    n = len(mat)
    print("   " + " ".join(f"{label(j):>4s}" for j in range(n)))
    for i, row in enumerate(mat):
        print(f"{label(i):>2s} " + " ".join(f"{x:>4d}" for x in row))

def display_max_flow(C: List[List[int]], R: List[List[int]]):
    n = len(C)
    print("⋆ Max flow display :")
    print("   " + " ".join(f"{label(j):>4s}" for j in range(n)))
    for i in range(n):
        cells = []
        for j in range(n):
            if C[i][j] > 0:
                used = C[i][j] - R[i][j]
                cells.append(f"{used}/{C[i][j]}")
            else:
                cells.append("0")
        print(f"{label(i):>2s} " + " ".join(f"{c:>4s}" for c in cells))
    total_flow = sum(C[0][j] - R[0][j] for j in range(n))
    print(f"\nValue of the max flow max = {total_flow}\n")

# --- 3) BFS + trace for FF ---
def bfs_with_parents(residual: List[List[int]], s: int, t: int
                    ) -> Tuple[Optional[List[int]], List[int]]:
    n = len(residual)
    parent = [-1]*n
    visited = [False]*n
    q = deque([s]); visited[s] = True

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

    for layer in layers[1:]:
        names = "".join(label(v) for v in layer)
        parts = "; ".join(f"Π({label(v)}) = {label(parent[v])}" for v in layer)
        print(f"{names} ; {parts}")

    if not visited[t]:
        return None, parent

    path = [t]
    while path[-1] != s:
        path.append(parent[path[-1]])
    path.reverse()
    return path, parent

# --- 4) Ford–Fulkerson trace ---
def ford_fulkerson_trace(C: List[List[int]], s: int, t: int
                        ) -> Tuple[int, List[List[int]]]:
    n = len(C)
    R = [row[:] for row in C]
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

        delta = min(R[path[i]][path[i+1]] for i in range(len(path)-1))
        chain = "".join(label(v) for v in path)
        print(f"Detection of an improving chain : {chain} with a flow {delta}.\n")

        for u, v in zip(path, path[1:]):
            R[u][v] -= delta
            R[v][u] += delta

        print("Modifications to the residual graph :")
        display_labeled_matrix(R)
        print()
        max_flow += delta
        iteration += 1

    return max_flow, R

# --- 5) Silent helper to compute max-flow ---
def find_augmenting_path(R: List[List[int]], s: int, t: int) -> Optional[List[int]]:
    n = len(R)
    parent = [-1]*n
    visited = [False]*n
    q = deque([s]); visited[s] = True
    while q:
        u = q.popleft()
        for v in range(n):
            if not visited[v] and R[u][v] > 0:
                parent[v] = u
                visited[v] = True
                q.append(v)
                if v == t:
                    break
        if visited[t]:
            break
    if not visited[t]:
        return None
    path = [t]
    while path[-1] != s:
        path.append(parent[path[-1]])
    return list(reversed(path))


def compute_max_flow(C: List[List[int]], s: int, t: int) -> int:
    R = [r[:] for r in C]
    flow = 0
    while True:
        path = find_augmenting_path(R, s, t)
        if not path:
            break
        delta = min(R[u][v] for u, v in zip(path, path[1:]))
        for u, v in zip(path, path[1:]):
            R[u][v] -= delta
            R[v][u] += delta
        flow += delta
    return flow

# --- 6) Bellman‐Ford trace for min‐cost ---
def display_bellman_snapshot(dist: List[float], pred: List[int], iteration: int):
    n = len(dist)
    print(f"Bellman-Ford iteration {iteration}:")
    print("   " + " ".join(f"{label(i):>6s}" for i in range(n)))
    print("d: " + " ".join(f"{int(dist[i]) if dist[i] != math.inf else '  INF':>6}" for i in range(n)))
    print("p: " + " ".join(f"{label(pred[i]) if pred[i] != -1 else '   -':>6}" for i in range(n)))
    print()

def bellman_ford_trace(n: int,
                       cost_res: List[List[int]],
                       R: List[List[int]],
                       s: int) -> Optional[List[int]]:
    dist = [math.inf]*n
    pred = [-1]*n
    dist[s] = 0
    for it in range(1, n):
        for u in range(n):
            for v in range(n):
                if R[u][v] > 0 and dist[u] + cost_res[u][v] < dist[v]:
                    dist[v] = dist[u] + cost_res[u][v]
                    pred[v] = u
        display_bellman_snapshot(dist, pred, it)
    return None if dist[-1] == math.inf else pred

# --- 7) Min‐cost successive‐shortest augmenting‐path ---
def min_cost_flow(n: int,
                  C: List[List[int]],
                  D: List[List[int]],
                  s: int,
                  t: int,
                  F: int) -> Tuple[int, int]:
    R = [r[:] for r in C]
    cost_res = [[0]*n for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if C[u][v] > 0:
                cost_res[u][v] = D[u][v]
                cost_res[v][u] = -D[u][v]

    flow = 0
    cost = 0
    remain = F
    iteration = 1
    while flow < F:
        print(f"⋆ Min-cost iteration {iteration}: remaining flow = {remain}")
        pred = bellman_ford_trace(n, cost_res, R, s)
        if pred is None:
            print("No augmenting path found; desired flow not reachable.")
            break

        # reconstruct path
        path = [t]
        while path[-1] != s:
            path.append(pred[path[-1]])
        path.reverse()

        delta = min(R[u][v] for u, v in zip(path, path[1:]))
        delta = min(delta, remain)
        inc_cost = sum(D[u][v] for u, v in zip(path, path[1:])) * delta

        print(f"Augmenting along path {''.join(label(v) for v in path)} with flow {delta}\n")
        for u, v in zip(path, path[1:]):
            R[u][v] -= delta
            R[v][u] += delta

        print("Modifications to the residual graph :")
        display_labeled_matrix(R)
        print()

        flow += delta
        cost += inc_cost
        remain -= delta
        iteration += 1

    return flow, cost

# --- 8) Push–Relabel trace ---
def push_relabel_trace(C: List[List[int]], s: int, t: int) -> Tuple[int, List[List[int]]]:
    n = len(C)
    R = [row[:] for row in C]
    height = [0]*n
    excess = [0]*n
    height[s] = n
    # Initial preflow from s
    for v in range(n):
        if C[s][v] > 0:
            flow = C[s][v]
            R[s][v] -= flow
            R[v][s] += flow
            excess[v] = flow
            excess[s] -= flow
            print(f"PUSH {flow} from {label(s)} to {label(v)} (initial preflow)")
    # List of active vertices
    active = deque([u for u in range(n) if u not in (s, t) and excess[u] > 0])
    iteration = 1
    while active:
        u = active[0]
        pushed = False
        # Try to push
        for v in range(n):
            if R[u][v] > 0 and height[u] == height[v] + 1:
                delta = min(excess[u], R[u][v])
                R[u][v] -= delta
                R[v][u] += delta
                excess[u] -= delta
                old_excess_v = excess[v]
                excess[v] += delta
                print(f"Iteration {iteration}: PUSH {delta} from {label(u)} to {label(v)}; excess[{label(u)}]={excess[u]}; excess[{label(v)}]={excess[v]}")
                if v not in (s, t) and old_excess_v == 0:
                    active.append(v)
                if excess[u] == 0:
                    active.popleft()
                pushed = True
                break
        if not pushed:
            # Relabel u
            min_h = min(height[v] for v in range(n) if R[u][v] > 0)
            old_h = height[u]
            height[u] = min_h + 1
            print(f"Iteration {iteration}: RELABEL {label(u)} from {old_h} to {height[u]}")
        iteration += 1
    max_flow = excess[t]
    print(f"\nValue of the max flow max = {max_flow}\n")
    return max_flow, R

def main():
    # compute project root as the parent of the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    tf = os.path.join(project_root, "test_files")

    while True:
        sel = input("Select problem number (1–10) or q to quit: ").strip()
        if sel.lower() == 'q':
            break
        if not sel.isdigit() or not (1 <= int(sel) <= 10):
            print("Please enter a number 1–10.")
            continue
        num = int(sel)
        fname = os.path.join(tf, f"example{num}.txt")

        # Determine algorithm choice based on problem number
        if 1 <= num <= 5:
            alg = input("Choose algorithm ([max] Ford-Fulkerson / [push] Push-Relabel): ").strip().lower()
            if alg.startswith('m'):  # guard against typo
                alg = 'max'
            if alg not in ('max', 'push'):
                print("Enter 'max' or 'push'.")
                continue
        else:
            alg = 'min'
            print(f"Automatically selecting Min-Cost for example{num}.txt")

        if not os.path.exists(fname):
            print(f"Missing file: {fname}\n")
            continue

        print()
        if alg == 'max':
            print(f"=== Ford–Fulkerson on example{num}.txt ===\n")
            C = read_capacity_file(fname)
            max_flow, R = ford_fulkerson_trace(C, 0, len(C)-1)
            display_max_flow(C, R)

        elif alg == 'push':
            print(f"=== Push–Relabel on example{num}.txt ===\n")
            C = read_capacity_file(fname)
            max_flow, R = push_relabel_trace(C, 0, len(C)-1)
            display_max_flow(C, R)

        else:  # min-cost
            print(f"=== Min-cost flow on example{num}.txt ===\n")
            n, C, D = read_min_cost_file(fname)
            print("⋆ Capacity matrix:")
            display_matrix(C)
            print("\n⋆ Cost matrix:")
            display_matrix(D)
            print()
            full = compute_max_flow(C, 0, n-1)
            F = full // 2
            print(f"Computed max flow = {full}, so target F = {F}\n")
            flow, total_cost = min_cost_flow(n, C, D, 0, n-1, F)
            print(f"Result: flow = {flow}, total cost = {total_cost}\n")
if __name__ == "__main__":
    main()