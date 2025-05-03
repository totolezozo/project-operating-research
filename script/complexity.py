import random
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Placeholder implementations; replace with your versions

def ford_fulkerson(capacity_matrix):
    """Compute max flow using Ford-Fulkerson (DFS augmenting path). Source=0, sink=n-1."""
    n = capacity_matrix.shape[0]
    # Residual capacity matrix
    residual = capacity_matrix.copy().astype(int)
    source, sink = 0, n-1
    parent = [-1] * n

    def dfs(u, visited):
        visited[u] = True
        if u == sink:
            return True
        for v in range(n):
            if not visited[v] and residual[u, v] > 0:
                parent[v] = u
                if dfs(v, visited):
                    return True
        return False

    max_flow = 0
    while True:
        visited = [False] * n
        parent = [-1] * n
        if not dfs(source, visited):
            break
        # Find minimum residual capacity along the path
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u, v])
            v = u
        # Update residual capacities
        v = sink
        while v != source:
            u = parent[v]
            residual[u, v] -= path_flow
            residual[v, u] += path_flow
            v = u
        max_flow += path_flow
    return max_flow


def push_relabel(capacity_matrix):
    """Compute max flow using Push-Relabel algorithm. Source=0, sink=n-1."""
    n = capacity_matrix.shape[0]
    source, sink = 0, n-1
    # Initialize residual capacities and flow
    residual = capacity_matrix.copy().astype(int)
    height = [0] * n
    excess = [0] * n
    height[source] = n
    # Preflow from source
    for v in range(n):
        if residual[source, v] > 0:
            excess[v] = residual[source, v]
            residual[v, source] = residual[source, v]
            residual[source, v] = 0
    # List of active vertices excluding source and sink
    active = [i for i in range(n) if i not in (source, sink) and excess[i] > 0]

    def push(u, v):
        send = min(excess[u], residual[u, v])
        residual[u, v] -= send
        residual[v, u] += send
        excess[u] -= send
        excess[v] += send

    def relabel(u):
        # Find minimum height of neighbor with available capacity
        min_height = float('inf')
        for v in range(n):
            if residual[u, v] > 0:
                min_height = min(min_height, height[v])
        if min_height < float('inf'):
            height[u] = min_height + 1

    idx = 0
    while idx < len(active):
        u = active[idx]
        old_height = height[u]
        # Try pushing to any neighbor
        for v in range(n):
            if residual[u, v] > 0 and height[u] == height[v] + 1:
                push(u, v)
                if excess[u] == 0:
                    break
        if excess[u] > 0:
            # Need to relabel
            relabel(u)
        if height[u] > old_height:
            # Move to front
            active.insert(0, active.pop(idx))
            idx = 0
        else:
            idx += 1
    return excess[sink]


def min_cost_flow(capacity_matrix, cost_matrix, target_flow):
    """Compute min-cost flow of value target_flow using Successive Shortest Path."""
    n = capacity_matrix.shape[0]
    source, sink = 0, n-1
    flow = 0
    cost = 0
    # Residual data structures
    capacity = capacity_matrix.copy().astype(int)
    cost = cost_matrix.copy().astype(int)
    # Initialize residual capacity and cost for reverse edges
    residual_cap = capacity.copy()
    residual_cost = cost.copy()
    # Add reverse edges with zero capacity and negative cost
    rev_cap = np.zeros_like(capacity)
    rev_cost = -cost

    # Combined residual capacities and costs
    R_cap = residual_cap + rev_cap
    R_cost = residual_cost + rev_cost

    # Potentials for reduced costs (Johnson's)
    potential = np.zeros(n, dtype=int)

    while flow < target_flow:
        # Bellman-Ford to find shortest paths in residual graph
        dist = [float('inf')] * n
        parent = [(-1, -1)] * n  # (u, direction): direction 0=forward,1=reverse
        dist[source] = 0
        in_queue = [False] * n
        queue = [source]
        in_queue[source] = True
        while queue:
            u = queue.pop(0)
            in_queue[u] = False
            for v in range(n):
                # Forward edge
                if R_cap[u, v] > 0:
                    rcost = R_cost[u, v] + potential[u] - potential[v]
                    if dist[u] + rcost < dist[v]:
                        dist[v] = dist[u] + rcost
                        parent[v] = (u, 0)
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True
                # Reverse edge
                if R_cap[v, u] > 0:
                    rcost = -R_cost[v, u] + potential[u] - potential[v]
                    if dist[u] + rcost < dist[v]:
                        dist[v] = dist[u] + rcost
                        parent[v] = (u, 1)
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True
        if parent[sink][0] == -1:
            break  # cannot send more flow
        # Update potentials
        for i in range(n):
            if dist[i] < float('inf'):
                potential[i] += dist[i]
        # Determine augment amount
        increment = target_flow - flow
        v = sink
        while v != source:
            u, direction = parent[v]
            if direction == 0:
                increment = min(increment, R_cap[u, v])
            else:
                increment = min(increment, R_cap[v, u])
            v = u
        # Apply augmentation
        v = sink
        while v != source:
            u, direction = parent[v]
            if direction == 0:
                R_cap[u, v] -= increment
                R_cap[v, u] += increment
                cost += increment * cost_matrix[u, v]
            else:
                R_cap[v, u] -= increment
                R_cap[u, v] += increment
                cost -= increment * cost_matrix[v, u]
            v = u
        flow += increment
    return flow, cost


def generate_random_problem(n):
    """Generate random capacity and cost matrices for n vertices."""
    C = np.zeros((n, n), dtype=int)
    D = np.zeros((n, n), dtype=int)
    # Fill roughly half of off-diagonal entries with random 1-100
    num_edges = int((n*(n-1))//4)
    edges = random.sample([(i, j) for i in range(n) for j in range(n) if i != j], num_edges)
    for i, j in edges:
        C[i, j] = random.randint(1, 100)
        D[i, j] = random.randint(1, 100)
    return C, D


def measure_time(func, *args, repeats=1):
    """Time a single call to func(*args)."""
    start = time.process_time()
    for _ in range(repeats):
        result = func(*args)
    end = time.process_time()
    return (end - start) / repeats, result


def run_experiments(n_values, trials=100):
    results = { 'FF': defaultdict(list), 'PR': defaultdict(list), 'MIN': defaultdict(list) }
    for n in n_values:
        print(f"→ Starting n={n} ({trials} trials)…", flush=True)
        for i in range(trials):
            if i % max(1, trials//10) == 0:
                print(f"   trial {i+1}/{trials}", end='\r', flush=True)
            C, D = generate_random_problem(n)
            t_ff, _ = measure_time(ford_fulkerson, C)
            results['FF'][n].append(t_ff)
            t_pr, _ = measure_time(push_relabel, C)
            results['PR'][n].append(t_pr)
            _, max_flow = measure_time(push_relabel, C)
            target = max_flow // 2
            t_min, _ = measure_time(min_cost_flow, C, D, target)
            results['MIN'][n].append(t_min)
        print()  # newline after each n
    print("✓ All experiments complete.")
    return results


def plot_scatter(results):
    """Plot scatter of runtimes for each algorithm."""
    for label, data in results.items():
        xs, ys = [], []
        for n, times in data.items():
            xs.extend([n] * len(times))
            ys.extend(times)
        plt.figure()
        plt.scatter(xs, ys, alpha=0.6)
        plt.title(f"Scatter of runtimes: {label}")
        plt.xlabel("n (vertices)")
        plt.ylabel("Time (s)")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.show()


def plot_worst_case(results):
    """Plot worst-case envelope on log-log scale and fit slope."""
    for label, data in results.items():
        ns = sorted(data.keys())
        max_times = [max(data[n]) for n in ns]
        # Fit line on log-log
        log_n = np.log(ns)
        log_t = np.log(max_times)
        coeff = np.polyfit(log_n, log_t, 1)
        slope = coeff[0]

        plt.figure()
        plt.plot(ns, max_times, 'o-', label='Worst-case')
        plt.plot(ns, np.exp(coeff[1]) * np.array(ns)**slope,
                 '--', label=f'Fit: n^{slope:.2f}')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"Worst-case envelope & fit: {label}")
        plt.xlabel("n (vertices)")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()


def plot_ratio(results, label1='FF', label2='PR'):
    """Plot ratio of worst-case runtimes: theta_label1/theta_label2."""
    ns = sorted(set(results[label1].keys()) & set(results[label2].keys()))
    ratios = []
    for n in ns:
        ratios.append(max(results[label1][n]) / max(results[label2][n]))

    plt.figure()
    plt.plot(ns, ratios, 'o-')
    plt.xscale('log')
    plt.title(f"Worst-case ratio: {label1}/{label2}")
    plt.xlabel("n (vertices)")
    plt.ylabel("Ratio of times")
    plt.grid(True, which="both", ls="--")
    plt.show()


if __name__ == '__main__':
    # for a quick smoke‐test use larger but still small instances:
    n_values = [10, 20, 40, 100, 400]
    trials = 100
    results = run_experiments(n_values, trials)
    plot_scatter(results)

