# Flow Algorithms Complexity Project

This repository contains multiple implementations and experiments for three core flow algorithms:

1. **Max-Flow** (Ford–Fulkerson / Edmonds–Karp)  
2. **Push–Relabel**  
3. **Min-Cost Flow** (Successive Shortest Path + Potentials)

## Project Structure

```

├── README.md                    
├── script/
│   ├── Main.py                  # interactive version with detailed logs
│   ├── complexity.py            # “basic” version without prints, sequential runs
│   └── complexity\_faster.py    # accelerated version (Dinic + multiprocessing)
│
├── test\-files/                 # input instances (graphs to solve)
│   ├──  from 1.txt to 5.txt     # instances for Max-Flow & Push–Relabel
│   └──  from 6.txt to 10.txt    # instances for Min-Cost Flow
│
├── traces/                     # textual outputs produced by complexity.py
│   ├── trace1-maxflow\.txt
│   ├── trace1-pushrelabel.txt
│   └── …
│
├── screenshots/                # screenshots of generated plots
│
└── faster/                     # work-in-progress optimized code (Cython/Numba)
├── flow\_cy.pyx                # Cython extension (in development)
├── setup.py                    # build script for Cython extension
└── main.py                     # driver to exercise `flow_cy` module

```



## ⚙️ Dependencies

### Basic version (`complexity.py`)
```bash
pip install numpy matplotlib
```

### Accelerated version (`complexity-faster.py`)

```bash
pip install numpy matplotlib tqdm
```

*Note: this version also uses standard-library modules `multiprocessing` and `heapq`.*

## How to Run

1. **Basic Complexity Measurement**

   ```bash
   cd script
   python complexity.py
   ```

   * Reads `test_files/1.txt` through `5.txt` for Max-Flow and Push-Relabel, then
     `test_files/6.txt` through `10.txt` for Min-Cost Flow.
   * Writes detailed traces into `traces/`.
   * Displays point-cloud scatter plots, worst-case envelopes (log-log), and ratios.

2. **Accelerated Measurement**

   ```bash
   cd script
   python complexity_faster.py
   ```

   * Same protocol (100 trials per size) using Dinic’s algorithm for Max-Flow
     and parallelizes trials across CPU cores.
   * Produces equivalent plots: scatter, envelope, and Max-Flow/Push-Relabel ratio.

3. **Cython-Backed Solver (Work in Progress)**

   ```bash
   cd faster
   py -3 -m pip install cython numpy
   py -3 setup.py build_ext --inplace
   python main.py
   ```

   * Compiles `flow_cy.pyx` into a native extension (`flow_cy.*.pyd`).
   * Demonstrates usage of `FlowSolver.edmonds_karp()`, `push_relabel()`,
     and `min_cost_flow()`.
   * **Note:** it's quite messy rn.

## File Format

* Each `test_files/*.txt` starts with an integer `n`, followed by two `n×n` matrices:

  * `C` (capacity matrix)
  * `D` (cost matrix)

* All scripts share the same function signatures:

  ```python
  MaxFlow(C) → int
  PushRelabel(C) → int
  MinCostFlow(C, D, target_flow) → (flow, cost)
  ```

---

## Output

* **Traces** (`traces/`): detailed step-by-step logs for each input and algorithm.
* **Screenshots** (`screenshots/`): visual examples of the scatter plots, envelopes,
  and ratio charts.

---

## Authors

* Thomas Chometon
* David Khersis
* Lilou Becker
* Thomas Grégroire
* Edgar Tromparent

Graph Theory Project – Complexity Analysis of Flow Algorithms

```

adjusting path to suit your setup should not be necessarry but try it if you need.
```
