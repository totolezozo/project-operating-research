# cython: boundscheck=False, wraparound=False, language_level=3
import numpy as np
cimport numpy as np
import heapq

cdef class FlowSolver:
    cdef int n
    # adjacency as 1-D buffers
    cdef np.int64_t[::1] head
    cdef np.int64_t[::1] to
    cdef np.int64_t[::1] cap
    cdef np.int64_t[::1] cost
    cdef np.int64_t[::1] nxt
    cdef np.int64_t[::1] rev
    cdef int edge_cnt

    def __cinit__(self, int n):
        cdef int size
        self.n = n

        # head array, initialized to -1
        cdef np.ndarray[np.int64_t, ndim=1] _head = np.empty(n, dtype=np.int64)
        _head.fill(-1)
        self.head = _head

        # reserve 4·n² slots for edges (forward+reverse)
        size = 4 * n * n
        self.to   = np.empty(size, dtype=np.int64)
        self.cap  = np.empty(size, dtype=np.int64)
        self.cost = np.empty(size, dtype=np.int64)
        self.nxt  = np.empty(size, dtype=np.int64)
        self.rev  = np.empty(size, dtype=np.int64)
        self.edge_cnt = 0

    cpdef void add_edge(self, int u, int v, long c, long w):
        cdef int e = self.edge_cnt
        # forward
        self.to[e]   = v
        self.cap[e]  = c
        self.cost[e] = w
        self.rev[e]  = e + 1
        self.nxt[e]  = self.head[u]
        self.head[u] = e
        e += 1
        # reverse
        self.to[e]   = u
        self.cap[e]  = 0
        self.cost[e] = -w
        self.rev[e]  = e - 1
        self.nxt[e]  = self.head[v]
        self.head[v] = e
        self.edge_cnt = e + 1

    cpdef long edmonds_karp(self, np.ndarray[np.int64_t, ndim=2] C not None):
        cdef int n = self.n
        cdef int source = 0, sink = n - 1
        cdef int u, v, e_, head_u, qi, qj
        cdef long inc, flow = 0

        # rebuild adjacency from C
        self.edge_cnt = 0
        self.head.fill(-1)
        for u in range(n):
            for v in range(n):
                if C[u, v] > 0:
                    self.add_edge(u, v, C[u, v], 0)

        # BFS buffers
        cdef np.ndarray[np.int64_t, ndim=1] _parent = np.empty(n, dtype=np.int64)
        cdef np.int64_t[::1] parent = _parent
        cdef np.ndarray[np.int64_t, ndim=1] _q = np.empty(n, dtype=np.int64)
        cdef np.int64_t[::1] q = _q

        while True:
            parent.fill(-1)
            qi = qj = 0
            parent[source] = source
            q[qj] = source; qj += 1

            while qi < qj and parent[sink] == -1:
                u = <int>q[qi]; qi += 1
                head_u = self.head[u]
                while head_u != -1:
                    e_ = head_u
                    v  = <int>self.to[e_]
                    if self.cap[e_] > 0 and parent[v] == -1:
                        parent[v] = e_
                        q[qj] = v; qj += 1
                        if v == sink:
                            break
                    head_u = self.nxt[head_u]

            if parent[sink] == -1:
                break

            # find bottleneck
            inc = 1 << 60
            v = sink
            while v != source:
                e_ = parent[v]
                if self.cap[e_] < inc:
                    inc = self.cap[e_]
                v = <int>self.to[self.rev[e_]]

            # augment
            v = sink
            while v != source:
                e_ = parent[v]
                self.cap[e_]   -= inc
                self.cap[self.rev[e_]] += inc
                v = <int>self.to[self.rev[e_]]
            flow += inc

        return flow

    cpdef long push_relabel(self, np.ndarray[np.int64_t, ndim=2] C not None):
        cdef int n = self.n
        cdef int source = 0, sink = n - 1
        cdef int u, v
        cdef long delta

        # residual as a 2D view
        cdef np.int64_t[:, :] res = C
        cdef np.int64_t[::1] height = np.zeros(n, dtype=np.int64)
        cdef np.int64_t[::1] excess = np.zeros(n, dtype=np.int64)

        height[source] = n
        for v in range(n):
            delta = res[source, v]
            if delta > 0:
                excess[v]    = delta
                res[v, source] = delta
                res[source, v] = 0

        active = [i for i in range(n) if i not in (source, sink) and excess[i] > 0]
        cdef int idx = 0
        cdef long min_h

        while idx < len(active):
            u = active[idx]
            old_h = height[u]

            # push
            for v in range(n):
                if res[u, v] > 0 and height[u] == height[v] + 1:
                    delta = res[u, v] if excess[u] > res[u, v] else excess[u]
                    res[u, v] -= delta
                    res[v, u] += delta
                    excess[u] -= delta
                    excess[v] += delta
                    if excess[u] == 0:
                        break

            # relabel
            if excess[u] > 0:
                min_h = 1 << 60
                for v in range(n):
                    if res[u, v] > 0 and height[v] < min_h:
                        min_h = height[v]
                height[u] = min_h + 1

            # discharge ordering
            if height[u] > old_h:
                active.insert(0, active.pop(idx))
                idx = 0
            else:
                idx += 1

        return excess[sink]

    cpdef tuple min_cost_flow(self,
                              np.ndarray[np.int64_t, ndim=2] C not None,
                              np.ndarray[np.int64_t, ndim=2] D not None,
                              long target):
        cdef int n = self.n
        cdef int source = 0, sink = n - 1
        cdef int u, v, e_, head_u
        cdef long flow = 0, total_cost = 0, inc
        cdef long INF = 1 << 60

        # rebuild adjacency
        self.edge_cnt = 0
        self.head.fill(-1)
        for u in range(n):
            for v in range(n):
                if C[u, v] > 0:
                    self.add_edge(u, v, C[u, v], D[u, v])

        cdef np.int64_t[::1] potential = np.zeros(n, dtype=np.int64)
        cdef np.int64_t[::1] dist      = np.empty(n, dtype=np.int64)
        cdef np.int64_t[::1] pv        = np.empty(n, dtype=np.int64)
        cdef np.int64_t[::1] pe        = np.empty(n, dtype=np.int64)
        cdef object tup
        cdef long d

        while flow < target:
            for u in range(n):
                dist[u] = INF
            dist[source] = 0

            heap = [(0, source)]
            while heap:
                tup = heapq.heappop(heap)
                d, u = tup
                if d > dist[u]:
                    continue
                head_u = self.head[u]
                while head_u != -1:
                    e_ = head_u
                    v  = <int>self.to[e_]
                    if self.cap[e_] > 0:
                        inc = d + self.cost[e_] + potential[u] - potential[v]
                        if inc < dist[v]:
                            dist[v] = inc
                            pv[v]   = u
                            pe[v]   = e_
                            heapq.heappush(heap, (inc, v))
                    head_u = self.nxt[head_u]

            if dist[sink] == INF:
                break

            for u in range(n):
                if dist[u] < INF:
                    potential[u] += dist[u]

            inc = target - flow
            v = sink
            while v != source:
                e_ = pe[v]
                if self.cap[e_] < inc:
                    inc = self.cap[e_]
                v = pv[v]

            v = sink
            while v != source:
                e_ = pe[v]
                self.cap[e_]            -= inc
                self.cap[self.rev[e_]] += inc
                total_cost             += inc * self.cost[e_]
                v = pv[v]

            flow += inc

        return flow, total_cost
