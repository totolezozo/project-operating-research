import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import heapq
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- 1) Vectorized random graph generator ---
def generate_random_problem(n, p=0.25):
    """Generate random capacity and cost matrices for n vertices with density p."""
    mask = (np.random.rand(n, n) < p)
    np.fill_diagonal(mask, False)
    C = (np.random.randint(1, 101, size=(n,n)) * mask).astype(int)
    D = (np.random.randint(1, 101, size=(n,n)) * mask).astype(int)
    return C, D

# --- 2) Dinic’s max‐flow ---
class Dinic:
    def __init__(self, n):
        self.N = n
        self.adj = [[] for _ in range(n)]
    def add_edge(self, u, v, w):
        self.adj[u].append([v, w, len(self.adj[v])])
        self.adj[v].append([u, 0, len(self.adj[u]) - 1])
    def max_flow(self, s, t):
        flow, INF = 0, 10**18
        N, adj = self.N, self.adj
        while True:
            level = [-1]*N; level[s] = 0
            q = [s]
            for u in q:
                for v, cap, _ in adj[u]:
                    if cap and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
            if level[t] < 0:
                break
            it = [0]*N
            def dfs(u, f):
                if u == t:
                    return f
                for i in range(it[u], len(adj[u])):
                    v, cap, rev = adj[u][i]
                    if cap and level[v] == level[u] + 1:
                        pushed = dfs(v, min(f, cap))
                        if pushed:
                            adj[u][i][1] -= pushed
                            adj[v][rev][1] += pushed
                            return pushed
                    it[u] += 1
                return 0
            while True:
                pushed = dfs(s, INF)
                if not pushed:
                    break
                flow += pushed
        return flow

def dinic_max_flow(C):
    """Wrap Dinic for capacity‐matrix C."""
    n = C.shape[0]
    D = Dinic(n)
    for u in range(n):
        for v, cap in enumerate(C[u]):
            if cap:
                D.add_edge(u, v, int(cap))
    return D.max_flow(0, n-1)

# --- 3) Basic Push-Relabel (we keep your implementation for comparison) ---
def push_relabel(C):
    n = C.shape[0]
    source, sink = 0, n-1
    residual = C.copy().astype(int)
    height = [0]*n; height[source] = n
    excess = [0]*n
    for v in range(n):
        if residual[source,v]:
            excess[v] = residual[source,v]
            residual[v,source] = residual[source,v]
            residual[source,v] = 0
    active = [i for i in range(n) if i not in (source,sink) and excess[i]>0]
    def push(u,v):
        send = min(excess[u], residual[u,v])
        residual[u,v] -= send; residual[v,u] += send
        excess[u] -= send; excess[v] += send
    def relabel(u):
        min_h = min([height[v] for v in range(n) if residual[u,v]>0], default=1e9)
        height[u] = min_h + 1
    idx=0
    while idx < len(active):
        u = active[idx]
        old_h = height[u]
        for v in range(n):
            if residual[u,v]>0 and height[u]==height[v]+1:
                push(u,v)
                if excess[u]==0:
                    break
        if excess[u]>0:
            relabel(u)
        if height[u] > old_h:
            active.insert(0, active.pop(idx))
            idx = 0
        else:
            idx += 1
    return excess[sink]

# --- 4) Min-Cost Flow via successive shortest path + potentials ---
def min_cost_flow(C, D, target):
    n = C.shape[0]
    # build adjacency lists for residual graph
    adj = [[] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if C[u,v]>0:
                # (to, cap, cost, rev_index)
                adj[u].append([v, C[u,v], D[u,v], len(adj[v])])
                adj[v].append([u, 0, -D[u,v], len(adj[u])-1])
    flow, cost = 0, 0
    potential = [0]*n
    while flow < target:
        dist = [float('inf')]*n; dist[0]=0
        parent = [(-1,-1)]*n  # (u, edge_index)
        hq = [(0,0)]
        while hq:
            d,u = heapq.heappop(hq)
            if d>dist[u]:
                continue
            for ei,(v,cap,cc,rev) in enumerate(adj[u]):
                if cap>0:
                    nd = d + cc + potential[u] - potential[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        parent[v] = (u, ei)
                        heapq.heappush(hq, (nd,v))
        if parent[n-1][0] < 0:
            break
        for i in range(n):
            if dist[i]<float('inf'):
                potential[i] += dist[i]
        inc = target - flow
        v = n-1
        while v!=0:
            u, ei = parent[v]
            inc = min(inc, adj[u][ei][1])
            v = u
        v = n-1
        while v!=0:
            u, ei = parent[v]
            edge = adj[u][ei]
            edge[1] -= inc
            adj[v][edge[3]][1] += inc
            cost += inc * edge[2]
            v = u
        flow += inc
    return flow, cost

# --- 5) Single trial + timing ------------------------------------
def single_trial(args):
    n, seed = args
    np.random.seed(seed)
    C, D = generate_random_problem(n, p=0.25)
    # 1) Dinic
    t0 = time.perf_counter()
    mf = dinic_max_flow(C)
    t_ff = time.perf_counter() - t0
    # 2) Push-Relabel
    t0 = time.perf_counter()
    pr = push_relabel(C)
    t_pr = time.perf_counter() - t0
    # 3) Min-Cost Flow (target = half of max-flow)
    target = mf // 2
    t0 = time.perf_counter()
    mc = min_cost_flow(C, D, target)
    t_min = time.perf_counter() - t0
    return n, t_ff, t_pr, t_min

# --- 6) Driver: parallel experiments + plotting -----------------
def run_and_plot(n_values, trials=100):
    tasks = [(n, i) for n in n_values for i in range(trials)]
    total = len(tasks)
    print(f"Launching {total} trials on {cpu_count()} cores…")

    data = {'FF':defaultdict(list), 'PR':defaultdict(list), 'MIN':defaultdict(list)}

    with Pool() as pool:
        # imap_unordered will yield results as they complete
        for n, t_ff, t_pr, t_min in tqdm(pool.imap_unordered(single_trial, tasks),
                                        total=total,
                                        desc="Overall progress",
                                        unit="trial"):
            data['FF'][n].append(t_ff)
            data['PR'][n].append(t_pr)
            data['MIN'][n].append(t_min)

    # scatter
    for label in data:
        plt.figure()
        xs = []; ys = []
        for n, times in data[label].items():
            xs += [n]*len(times)
            ys += times
        plt.scatter(xs, ys, alpha=0.6)
        plt.xscale('log'); plt.yscale('log')
        plt.title(f"Scatter: {label}"); plt.xlabel("n"); plt.ylabel("time (s)")
        plt.grid(True, which='both', ls='--')
    # worst‐case envelope + fit
    for label in data:
        ns = sorted(data[label].keys())
        max_t = [max(data[label][n]) for n in ns]
        coeff = np.polyfit(np.log(ns), np.log(max_t), 1)
        slope = coeff[0]
        plt.figure()
        plt.plot(ns, max_t, 'o-', label='max')
        plt.plot(ns, np.exp(coeff[1]) * np.array(ns)**slope,
                 '--', label=f"n^{slope:.2f}")
        plt.xscale('log'); plt.yscale('log')
        plt.title(f"Worst-case & fit: {label}")
        plt.legend(); plt.grid(True, which='both', ls='--')
    # ratio FF/PR
    ns = sorted(set(data['FF']) & set(data['PR']))
    ratios = [max(data['FF'][n]) / max(data['PR'][n]) for n in ns]
    plt.figure()
    plt.plot(ns, ratios, 'o-'); plt.xscale('log')
    plt.title("Worst-case ratio FF/PR"); plt.xlabel("n"); plt.ylabel("ratio")
    plt.grid(True, which='both', ls='--')

    plt.show()

if __name__ == '__main__':
    # final experiment sizes
    n_values = [10, 20, 40, 100, 400, 1000]
    run_and_plot(n_values, trials=100)
