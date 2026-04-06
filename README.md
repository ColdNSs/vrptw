# Vehicle Routing Problem with Time Windows: Bi-Level Optimization

A VRPTW solver inspired by **MMOEA-DL** (IEEE TEVC 2025) — a deep reinforcement learning assisted multimodal multi-objective bi-level optimization method for multi-robot task allocation.

---

## 📌 Project Goal

Implement a bi-level optimization solver for the **Vehicle Routing Problem with Time Windows (VRPTW)**:

- **Upper level**: Assign customers to vehicle routes (task allocation)
- **Lower level**: Sequence customers within each route (TSP / path planning)

Following the MMOEA-DL paradigm, we combine evolutionary exploration at the upper level with learned (DRL) and local search (LNS) exploitation at the lower level.

---

## 📊 Benchmark

- **Solomon-100** instances (`data/benchmarks/solomon-100/`)
- Three problem families: `C` (clustered), `R` (random), `RC` (mixed)
- Metrics: total distance, number of vehicles, solution feasibility

---

## 🛤️ Development Roadmap

### Phase 1 — Baseline Framework ✅
- [x] Solomon-100 benchmark parser
- [x] Euclidean distance matrix
- [x] Greedy solver (nearest-neighbor + feasibility checks)
- [x] Evaluation harness (metrics: route distance, total distance, finish time, makespan, time window penalties, etc.)

### Phase 2 — Bi-Level Structure
- [x] `Individual` class handles two separate concerns:
  - **Upper solver**: Genetic algorithm / differential evolution for customer-to-route assignment
  - **Lower solver**: Sequence optimization within each route (replaces greedy)
- [ ] Proper bi-level fitness evaluation

### Phase 3 — Multimodal Multi-Objective
- [ ] Multi-objective fitness: minimize total distance + total tardiness
- [ ] MMODE_CSCD crowding distance for diversity preservation
- [ ] Pareto front extraction and visualization

### Phase 4 — DRL Lower-Level Solver (Nazari et al. approach)
- [ ] Encoder-decoder with attention mechanism (GRU decoder)
- [ ] Actor-Critic training on Solomon instances
- [ ] End-to-end inference: given customer coordinates → optimal visiting order
- [ ] Integration as the lower-level solver in the bi-level framework

### Phase 5 — LNS Post-Optimization
- [ ] Destroy operator (random removal of customers from route)
- [ ] Repair operator (re-insertion with feasibility checks)
- [ ] Iterated LNS for local refinement of final-generation solutions

### Phase 6 — Benchmarking & Analysis
- [ ] Compare against Solomon best-known solutions
- [ ] Ablation study: contribution of DRL vs LNS vs evolutionary components
- [ ] Generalization test: train on one instance family, test on others

---

## 📂 Repository Structure

```
vrptw/
├── data/
│   └── benchmarks/solomon-100/   # Solomon benchmark files
├── src/
│   ├── main.py                   # Entry point
│   ├── parser.py                 # Solomon instance parser
│   ├── solver.py                 # GreedySolver (Phase 1 baseline)
│   └── utils.py                  # Distance matrix calculation
├── environment.yml               # Conda environment spec
└── README.md
```

---

## 🔬 Reference

**MMOEA-DL**: Fan et al., "A Deep Reinforcement Learning-Assisted Multimodal Multi-Objective Bi-Level Optimization Method for Multi-Robot Task Allocation," *IEEE TEVC*, 2025.

Key ideas adapted:
- Bi-level → single-level transformation via DRL end-to-end lower-level solver
- Evolutionary algorithm (DE-based) for upper-level task allocation
- LNS destroy-repair for final route refinement
- Multimodal objectives for multiple Pareto-equivalent solutions
