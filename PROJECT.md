# VRPTW Project — Working Notes

*Flavilar's working context for this project. Read before any session.*

---

## What This Project Is

**Goal**: Build a solver for the **Vehicle Routing Problem with Time Windows (VRPTW)** using a bi-level optimization approach inspired by the **MMOEA-DL** paper (IEEE TEVC 2025).

**Repo**: https://github.com/ColdNSs/vrptw
**Owner**: ColdNSs (Lazarus)
**My access**: Fork at https://github.com/flavilar/vrptw — submit PRs to `ColdNSs/vrptw`

---

## The MMOEA-DL Paper (Reference)

**Full citation**: Fan et al., "A Deep Reinforcement Learning-Assisted Multimodal Multi-Objective Bi-Level Optimization Method for Multi-Robot Task Allocation," *IEEE Transactions on Evolutionary Computation*, 2025.

**PDF**: `/home/node/.openclaw/media/inbound/A_Deep_Reinforcement_Learning_Assisted_Multimodal_Multi_Obje---7b6474de-df29-44db-9d92-095679994806.pdf`

### Core Architecture

| Level | Problem | Method |
|---|---|---|
| **Upper** | Task allocation (which robot does which tasks) | MMODE_CSCD — multimodal multi-objective DE |
| **Lower** | Path planning per robot (TSP) | DRL end-to-end (Actor-Critic, Nazari et al.) + LNS |

### Key Insight
- Bi-level optimization is computationally expensive (nested structure)
- DRL end-to-end for the lower level **transforms the bi-level problem into single-level** — the evolutionary algorithm only needs to evolve task allocations; routes are computed fast via the trained DRL
- LNS destroy-repair operators provide local refinement on final solutions

### Lower-Level DRL Details
- Encoder-decoder architecture
- Input: customer coordinates → embedding via linear layer
- Decoder: GRU with attention mechanism over all customers
- Output: permutation ( visiting order)
- Training: Actor-Critic (REINFORCE baseline)
- Loss: expected tour length (minimize)

### LNS Details
- **Destroy operator**: randomly remove I task points from a route
- **Repair operator**: re-insert removed points in all possible positions, pick the best
- Iterate for Gmax generations

---

## Current Project State

### Phase 1 — Baseline Framework ✅
- `src/parser.py`: Solomon-100 benchmark parser (depot + customer nodes with time windows)
- `src/utils.py`: Euclidean distance matrix calculator
- `src/solver.py`: `GreedySolver` — nearest-neighbor heuristic with capacity + time window feasibility checks
- `data/benchmarks/solomon-100/`: 56 Solomon instances (C, R, RC families)
- `environment.yml`: Conda environment spec
- `src/main.py`: Entry point (loads instance, runs greedy solver, prints routes)

### GreedySolver Logic
1. Start new route at depot
2. Repeatedly pick the nearest unvisited customer that fits (capacity + time window)
3. If no customer fits, close route (return to depot)
4. Repeat until all customers assigned
5. Return list of `Route` objects

### `Route` class
- `sequence`: list of `Node` objects (depot first)
- `load`: cumulative demand (checked against `capacity`)
- `time`: current time (wait for ready_time, add service_time + travel)
- `cost`: total distance traveled
- `is_feasible(node)`: checks capacity + arrival time ≤ due_date
- `close_route()`: attempts to return to depot

### `Node` class
- `id`, `x`, `y`, `demand`, `ready_time`, `due_date`, `service_time`
- Implements `__index__` so Node objects can index numpy arrays directly

### `SolomonInstance` class
- `name`, `num_vehicles`, `capacity`, `nodes` (list, index 0 = depot)

### Solomon-100 Benchmark
- 56 instances: C1xx (clustered), R1xx (random), RC1xx (mixed)
- Depot at index 0, 100 customers per instance
- Time windows: `[ready_time, due_date]`
- Vehicles have identical capacity and max duration

---

## Development Roadmap (from PR #1)

| Phase | Description | Status |
|---|---|---|
| 1 | Baseline framework | ✅ Done |
| 2 | Bi-level structure refactor | 🔲 Todo |
| 3 | DRL lower-level solver (Nazari et al.) | 🔲 Todo |
| 4 | LNS post-optimization | 🔲 Todo |
| 5 | Multimodal multi-objective extension | 🔲 Todo |
| 6 | Benchmarking & ablation | 🔲 Todo |

---

## Working with This Project

### Environment Setup
```bash
cd /home/node/.openclaw/workspace/vrptw
conda env create -f environment.yml
conda activate vrptw
python src/main.py
```

### Git Workflow
1. Work on the fork: `https://github.com/flavilar/vrptw`
2. Keep `upstream` tracking `ColdNSs/vrptw`
3. Create PRs against `ColdNSs/vrptw:main`
4. **Never push directly to upstream**

### Run Greedy Baseline
```bash
python src/main.py
```

---

## Notes for Future Sessions

- **First thing**: read this file before touching any project code
- **PRs**: always fork-and-PR workflow, never push directly
- **Solomon format**: txt files with VE...ICLE and CU...STOMER section headers
- **DRL approach**: Nazari et al. (2018) encoder-decoder with attention — this is the reference for the lower-level solver
- **LNS**: destroy (random removal) + repair (greedy re-insertion) — reference for Phase 4
- **Key design decision to make**: how to handle time windows in the DRL lower-level solver (the paper's DRL was for TSP without time windows — VRPTW adds feasibility constraints)

---

## Project-Specific Conventions (to be updated as we go)

- Python, no framework specified yet
- Solomon-100 benchmark confirmed
- Bi-level structure confirmed (GA upper + DRL+LNS lower)
- Multimodal multi-objective is the stretch goal
