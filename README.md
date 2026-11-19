# Vertigo CVRP demo

Branch-cut-and-price for the capacitated vehicle routing problem (CVRP) with
[Vertigo.jl](https://github.com/nablarise/Vertigo.jl). The pricing is the bucket
graph labeling algorithm of
[bucket-graph-spprc](https://github.com/spoorendonk/bucket-graph-spprc), called
through a small C shim.

- The model is an arc-flow CVRP decomposed with Dantzig-Wolfe. The master covers
  each customer once. One subproblem stands for the identical vehicles: a column is
  a route, used 0 to `fleet` times.
- Pricing runs the bucket graph, with load as the main resource and optional
  ng-routes.
- Cuts are optional: rounded capacity cuts, and k-path cuts in their robust form
  (`cuts.jl`).

## Requirements

- Julia 1.11 or later.
- A C++23 compiler (Clang 18 or later, GCC 14 or later), and git.
- A checkout of Vertigo.jl next to this one, in `../Vertigo`. `Project.toml` points
  at it by path.

## Setup

```sh
./deps/build.sh                               # bucket-graph-spprc + the C shim
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`deps/build.sh` clones bucket-graph-spprc at a pinned commit and compiles
`capi/bgspprc_c.cpp` into `deps/build/libbgspprc_c.dylib` (`.so` on Linux). To use
another build, set `BGSPPRC_C_LIB`.

## Run

```sh
julia --project=. cvrp.jl instances/A-n32-k5.vrp --ng 8 --max-paths 30 --capacity-cuts
julia --project=. cvrp.jl --help
```

Each run solves the instance once per node selection strategy: depth-first,
best-first, and random.

| Option | What it does |
|---|---|
| `--capacity-cuts` | Separate rounded capacity cuts |
| `--kpath-cuts` | Separate k-path cuts, where k(S) is the bin-packing number of S |
| `--cut-rounds N` | Cut rounds per node (default 20) |
| `--edge-branching` | Branch on edge flows `x_ij + x_ji` instead of arcs |
| `--customers N`, `--fleet K` | Keep the first N customers; cap the routes at K |
| `--ng`, `--max-paths`, `--steps`, `--mono`, ... | Pricing options of `bgspprc-solve` |

## Results

These are for A-n32-k5, whose optimum is 784 (CVRPLIB). The settings are
`--ng 8 --max-paths 30`, on an Apple M3 Pro:

| Cuts | Root bound | Result |
|---|---|---|
| none | 758.0 | not solved in 300 s; gap 3.7% at best |
| capacity + k-path | 783.2 | optimal 784 in 11 nodes, 20 to 28 s |

No k-path cut is found on A-n32-k5. The demands there are small compared with the
capacity, so the bin-packing number of a set never exceeds `⌈q(S)/Q⌉`.

## Checking a result

`tools/reference.jl` enumerates every elementary, capacity-feasible route, then
solves the set-partitioning LP and IP with HiGHS. This gives a reference that does
not depend on Vertigo, for small instances (2^customers subsets):

```sh
julia --project=. tools/reference.jl instances/A-n32-k5.vrp 12 2   # LP 410.5, IP 416
```

## Workarounds and known limits

| What | Why | Tracked in |
|---|---|---|
| `set_route_bounds!` overwrites the convexity bounds after the decomposition | Vertigo fixes every subproblem to exactly one column | Vertigo.jl#84 |
| The pricer returns the route value without `fk` and the convexity duals as its dual bound | Vertigo adds them itself; the reduced cost there gives an invalid bound and wrong optima | Vertigo.jl#104 |
| `merge_edge_variables!` writes into Vertigo's variable mapping | Vertigo has no API to map several subproblem variables to one original variable | Vertigo.jl#105 |
| HiGHS is pinned to 1.24 | With HiGHS.jl 1.27.0 (HiGHS_jll 1.16.0), the restricted-master MIP reports a wrong objective value. Vertigo takes it as the incumbent value and closes the tree on a false optimum. | — |
| Decomposing large instances is slow | AffExpr terms are splatted in the DML | Vertigo.jl#99 |
| Only robust cuts | Vertigo cannot express cuts whose coefficient depends on the whole route (rank-1, strong k-path, route-load knapsack) | — |
| `capi/bgspprc_c.cpp` is our own C shim | bucket-graph-spprc has no C API upstream | — |
