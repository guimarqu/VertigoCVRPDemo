using JuMP
using HiGHS
using Vertigo
using Random
import MathOptInterface as MOI

include("bgspprc_c.jl")
include("cuts.jl")

const CG = Vertigo.ColGen

# Phase markers with time since start, flushed so a long set-up shows where it is.
const T_START = time()
const QUIET = Ref(false)   # --csv: nothing but the CSV on stdout
progress(msg) = QUIET[] || (println("[", lpad(round(time() - T_START; digits=1), 6), "s] ", msg); flush(stdout))

# ── Instance ──────────────────────────────────────────────────────────────────

# Node 1 is the depot, nodes 2:n+1 the customers (CVRPLIB order).
struct CVRPInstance
    name::String
    fleet::Int
    capacity::Float64
    demand::Vector{Float64}
    dist::Matrix{Float64}
end

n_customers(inst::CVRPInstance) = length(inst.demand) - 1

"""
    parse_cvrp(path; customers = nothing) -> CVRPInstance

Read a CVRPLIB `.vrp` file (EUC_2D, depot first), keeping the first `customers`
customers. Distances are Euclidean rounded to the nearest integer. The fleet
limit is `fleet`, else the `-kN` of the instance name, else unlimited.
"""
function parse_cvrp(
    path::AbstractString; customers::Union{Nothing,Int}=nothing, fleet::Union{Nothing,Int}=nothing
)
    lines = strip.(readlines(path))
    capacity = parse(Float64, last(split(lines[findfirst(startswith("CAPACITY"), lines)], ':')))
    section(name) = findfirst(startswith(name), lines) + 1
    n = parse(Int, last(split(lines[findfirst(startswith("DIMENSION"), lines)], ':')))
    rows(name) = [parse.(Float64, split(l)) for l in lines[section(name):(section(name)+n-1)]]
    coords, dem = rows("NODE_COORD_SECTION"), rows("DEMAND_SECTION")
    keep = customers === nothing ? n : customers + 1
    x, y = [r[2] for r in coords[1:keep]], [r[3] for r in coords[1:keep]]
    dist = [round(hypot(x[i] - x[j], y[i] - y[j])) for i in 1:keep, j in 1:keep]
    name = first(splitext(basename(path)))
    k = match(r"-k(\d+)", name)
    return CVRPInstance(
        name, something(fleet, k === nothing ? keep - 1 : parse(Int, k[1])), capacity,
        [r[2] for r in dem[1:keep]], dist
    )
end

# ── Pricing oracle ────────────────────────────────────────────────────────────

# Stands in for the MOI model of the pricing subproblem: Vertigo writes reduced
# arc costs into it, pricing runs the bucket graph labeling algorithm.
mutable struct CVRPPricer
    solver::BGSPPRCC.Solver
    arc_vars::Vector{MOI.VariableIndex}   # arc_vars[a]: subproblem variable of arc a
    arc_of::Dict{MOI.VariableIndex,Int}
    arc_cost::Vector{Float64}             # current reduced cost of every arc
    depot_arcs::Vector{Int}               # arcs leaving the source: every route uses exactly one
    threshold::Float64                    # pricing threshold θ of the solver (--theta)
    totals::Dict{Symbol,Float64}          # solver statistics summed over all pricing calls
end

CVRPPricer(solver, arc_vars, depot_arcs, threshold) = CVRPPricer(
    solver, arc_vars, Dict(v => a for (a, v) in enumerate(arc_vars)), zeros(length(arc_vars)),
    depot_arcs, threshold, Dict{Symbol,Float64}()
)

# Counters add up over the calls; sizes are those of the last call.
const LAST_STATS = (:n_buckets, :n_fixed_buckets, :label_state_bytes, :eliminated_bucket_arcs)

function accumulate!(totals, st::BGSPPRCC.Stats, n_paths)
    for f in fieldnames(BGSPPRCC.Stats)
        v = Float64(getfield(st, f))
        totals[f] = f in LAST_STATS ? v : get(totals, f, 0.0) + v
    end
    totals[:calls] = get(totals, :calls, 0.0) + 1
    totals[:paths] = get(totals, :paths, 0.0) + n_paths
    return totals
end

function MOI.modify(
    p::CVRPPricer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    change::MOI.ScalarCoefficientChange{Float64}
)
    p.arc_cost[p.arc_of[change.variable]] = change.new_coefficient
    return nothing
end

function CG.get_pricing_subprob_optimizer(::CG.ExactStage, sp::CG.PricingSubproblem{<:CVRPPricer})
    return CG.moi_pricing_sp(sp)
end

function CG.optimize_pricing_problem!(
    ws::CG.ColGenWorkspace, sp_id, ::CG.PricingSubproblem, p::CVRPPricer,
    mast_dual_sol, _stab_changes_mast_dual_sol
)
    # Vertigo logs with plain `println`; stdout is block-buffered unless it is a terminal,
    # so without this the iteration log only shows up once the run ends.
    flush(stdout)
    decomp = ws.decomp
    ν = CG._get_convexity_dual(mast_dual_sol, convexity_lb_pairs(decomp), sp_id) +
        CG._get_convexity_dual(mast_dual_sol, convexity_ub_pairs(decomp), sp_id)
    fk = subproblem_fixed_cost(decomp, sp_id)

    # The fixed cost and the convexity dual are paid once per route, on its departure from the
    # depot, so that the cost of a path is the reduced cost of its column and θ applies as is.
    arc_cost = copy(p.arc_cost)
    arc_cost[p.depot_arcs] .+= fk - ν
    θ = p.threshold
    paths, exact = BGSPPRCC.solve!(p.solver, arc_cost, θ)
    accumulate!(p.totals, BGSPPRCC.stats(p.solver), length(paths))

    # Only the exact stage proves optimality: with no route below θ, θ is a valid lower bound
    # and otherwise the best route is the optimum. A heuristic stage gives columns, no bound.
    best = isempty(paths) ? θ : minimum(path -> path.reduced_cost, paths)
    sp_obj = best
    # Vertigo adds fk and the convexity duals itself: the bound is the route's value without
    # them (Vertigo.jl#104).
    sp_bound = exact ? best - fk + ν : -Inf
    columns = CG.PricingPrimalSolution[]
    for path in paths
        counts = Dict{MOI.VariableIndex,Float64}()
        for a in path.arcs
            v = p.arc_vars[a]
            counts[v] = get(counts, v, 0.0) + 1.0
        end
        rc = path.reduced_cost
        sol = Vertigo.Reformulation._SpSolution(sp_id, rc, [(v, c) for (v, c) in counts])
        push!(columns, CG.PricingPrimalSolution(sp_id, sol, true))
    end
    return CG.PricingSolution(false, false, sp_obj, sp_bound, columns)
end

# ── Model ─────────────────────────────────────────────────────────────────────

# The `ng` closest customers of v (itself included), by distance.
function ng_neighbors(inst::CVRPInstance, v::Int, ng::Int)
    others = sort(setdiff(2:(n_customers(inst)+1), v); by=u -> (inst.dist[v, u], u))
    return vcat(v, first(others, ng - 1))
end

"""
    merge_edge_variables!(decomp, sp_id, arcs, arc_vars, x, sink)

Edge branching (Costa et al. 2019, 3.3.1): the instance is undirected, so the flow of the edge
{i, j} is x_ij + x_ji. Vertigo branches on original variables and sums the subproblem variables
mapped to the same one, so both directions of an edge are mapped to the original variable of
the arc (min, max); the sink counts as the depot. The branching constraint, the dual added to
the arc costs and the projection of the master solution then all concern the edge.

Vertigo has no API for this yet (Vertigo.jl#105): this writes into its mapping.
"""
function merge_edge_variables!(decomp, sp_id, arcs, arc_vars, x, sink)
    inverse = decomp.mapping.inverse
    node(v) = v == sink ? 1 : v
    for (a, sp_var) in zip(arcs, arc_vars)
        u, v = node(a[1]), node(a[2])
        # The arc of the edge that goes from the lower to the higher vertex (or out of the depot).
        canonical = u < v ? a : (a[2] == sink ? (1, a[1]) : (a[2], a[1]))
        canonical == a && continue
        inverse[(sp_id, sp_var)] = x[canonical...]
    end
    return nothing
end

"""
    build_cvrp_model(inst; ng = 0, bucket_steps = (10.0, 1.0), xi = 25.0, max_paths = 1, theta = -1e-6, ...) -> (decomp, cut_graph)

Arc-flow CVRP decomposed with Dantzig-Wolfe over graph vertex 1 = depot out,
2:n+1 = customers, n+2 = depot in. The master covers each customer exactly
once; a single subproblem (identical vehicles, 0 to `fleet` routes) owns the
arc variables and flow conservation. Capacity is not a constraint: the pricing
oracle enforces it as the bucket graph's main resource, with ng-route
relaxation of neighborhood size `ng` (0 = off). The remaining keywords are the flags of
`bgspprc-solve`, with its defaults; `theta` is the threshold θ of bgspprc, the reduced cost below which a route is
returned by pricing. `cut_graph` is what the cut separator of `cuts.jl` reads.
"""
function build_cvrp_model(
    inst::CVRPInstance; ng::Int=0, bucket_steps=(10.0, 1.0), xi::Float64=25.0, max_paths::Int=1, theta::Float64=-1.0e-6,
    bidirectional::Bool=true, stage::Symbol=:exact, auto_steps::Bool=false,
    parallel::Bool=true, parallel_bidir::Bool=true, edge_branching::Bool=false
)
    n = n_customers(inst)
    sink = n + 2
    node(v) = v == sink ? 1 : v
    dem(v) = inst.demand[node(v)]
    cost(a) = inst.dist[node(a[1]), node(a[2])]

    arcs = Tuple{Int,Int}[]
    for j in 2:(n+1)
        push!(arcs, (1, j), (j, sink))
        for i in 2:(n+1)
            i != j && dem(i) + dem(j) <= inst.capacity && push!(arcs, (i, j))
        end
    end
    progress("pricing graph: $(length(arcs)) arcs")
    inn = [Int[] for _ in 1:sink]
    out = [Int[] for _ in 1:sink]
    for (i, j) in arcs
        push!(out[i], j)
        push!(inn[j], i)
    end

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:sink, j in 1:sink; (i, j) in arcs], Bin)
    # Constraints are indexed 1:n (customer c is vertex c + 1).
    @constraint(model, cover[c in 1:n], sum(x[j, c+1] for j in inn[c+1]) == 1)
    @constraint(model, flow[c in 1:n], sum(x[j, c+1] for j in inn[c+1]) == sum(x[c+1, j] for j in out[c+1]))
    @objective(model, Min, sum(cost(a) * x[a...] for a in arcs))

    progress("JuMP model built")
    decomp, sp_map = @dantzig_wolfe model begin
        x[_, _] => subproblem(1)
        flow[_] => subproblem(1)
        cover[_] => master()
    end
    sp_id = sp_map[1]
    # Vertigo fixes every subproblem to 1 <= sum(lambda) <= 1 (Vertigo.jl#84).
    set_route_bounds!(decomp, sp_id, 0.0, Float64(inst.fleet))

    progress("Dantzig-Wolfe decomposition done")
    # Pricing graph: load is the single main resource, in [0, capacity] everywhere.
    solver = BGSPPRCC.Solver(;
        n_vertices=sink, source=1, sink,
        arc_from=first.(arcs), arc_to=last.(arcs), arc_cost=cost.(arcs),
        arc_resource=[[dem(j) for (_, j) in arcs]],
        vertex_lb=[zeros(sink)], vertex_ub=[fill(inst.capacity, sink)],
        ng=ng > 0 ? [2 <= v <= n + 1 ? ng_neighbors(inst, v, ng) : [v] for v in 1:sink] : nothing,
        bucket_steps, xi, max_paths, bidirectional, stage, auto_steps, parallel, parallel_bidir
    )

    progress("bucket graph built")
    arc_vars = [Vertigo.Reformulation.mapped_subproblem_var(decomp, x[a...])[2] for a in arcs]
    edge_branching && merge_edge_variables!(decomp, sp_id, arcs, arc_vars, x, sink)
    pricer = CVRPPricer(solver, arc_vars, findall(a -> a[1] == 1, arcs), theta)
    pricer.totals[:n_verts] = sink
    pricer.totals[:n_arcs] = length(arcs)
    sp_models(decomp)[sp_id] = pricer
    return decomp, CVRPCutGraph(inst, arcs, decomp, sp_id, arc_vars, sink)
end

# Overwrites the convexity bounds of `sp_id` in the master and in the decomposition,
# through Vertigo internals, until Vertigo.jl#84 lets the DML set them.
function set_route_bounds!(decomp, sp_id, lb, ub)
    m = master_model(decomp)
    with_typed_ci(ci -> MOI.set(m, MOI.ConstraintSet(), ci, MOI.GreaterThan(lb)), decomp.convexity_lb[sp_id])
    with_typed_ci(ci -> MOI.set(m, MOI.ConstraintSet(), ci, MOI.LessThan(ub)), decomp.convexity_ub[sp_id])
    old = decomp.subproblems[sp_id]
    decomp.subproblems[sp_id] = Vertigo.Reformulation.SubproblemData(
        old.variables, old.original_costs, old.coupling_coeffs, old.fixed_cost, lb, ub)
    return nothing
end

# ── Node selection ────────────────────────────────────────────────────────────

# A user-defined node selection: best-first on a priority drawn at random when the node is
# inserted. It reads nothing from the node, so it is the baseline the other strategies must beat.
struct RandomStrategy <: Vertigo.TreeSearch.AbstractBestFirstStrategy
    rng::Random.Xoshiro
end
RandomStrategy(seed::Integer) = RandomStrategy(Random.Xoshiro(seed))
Vertigo.TreeSearch.get_priority(s::RandomStrategy, ::Vertigo.TreeSearch.SearchNode) = rand(s.rng)

# The strategies of a benchmark run, in order. Built per run so each gets a fresh generator.
node_strategies(seed::Integer) = [
    "depth-first" => Vertigo.TreeSearch.DepthFirstStrategy(),
    "best-first" => Vertigo.TreeSearch.DualBoundBestFirstStrategy(),
    "random" => RandomStrategy(seed),
]

# ── Solve ─────────────────────────────────────────────────────────────────────

"Returns `(out, totals)`: the branch-and-price output and the pricing statistics summed over all calls."
function solve_cvrp(
    inst::CVRPInstance; node_limit::Int=typemax(Int), time_limit::Float64=180.0,
    silent::Bool=false, node_select_strategy=Vertigo.TreeSearch.DepthFirstStrategy(),
    cg_log_level::Int=0, log_level::Int=1, capacity_cuts::Bool=false, kpath_cuts::Bool=false,
    cut_rounds::Int=20, kwargs...
)
    decomp, cut_graph = build_cvrp_model(inst; kwargs...)
    sep = capacity_cuts || kpath_cuts ? CVRPCutSeparator(cut_graph; capacity_cuts, kpath_cuts) : nothing
    progress("starting branch-and-price")
    config = BranchCutPriceConfig(;
        colgen=ColGenConfig(smoothing_alpha=0.0, silent=silent),
        cg_log_level=silent ? 0 : cg_log_level,
        log_level=silent ? 0 : log_level,
        node_limit, time_limit, strategy=node_select_strategy, rmp_time_limit=10.0,
        separator=sep,
        max_cut_rounds=sep === nothing ? 0 : cut_rounds, min_gap_improvement=1e-4
    )
    out = run_branch_and_price(BranchCutPriceWorkspace(decomp, config))
    totals = only(values(sp_models(decomp))).totals
    if sep !== nothing
        totals[:capacity_cuts] = sep.counts[:capacity]
        totals[:kpath_cuts] = sep.counts[:kpath]
        totals[:cut_rounds] = sep.counts[:rounds]
    end
    return out, totals
end

# ── Command line ──────────────────────────────────────────────────────────────

# name => (kind, description). Kinds: a type to parse, :flag (no value) or :steps ("S1,S2").
# The bgspprc flags keep the names and defaults of `bgspprc-solve`.
const CLI_OPTIONS = [
    :customers => (Int, "keep the first N customers (default: all)"),
    :fleet => (Int, "max routes (default: the -kN of the file name, else unlimited)"),
    :node_limit => (Int, "branch-and-price nodes to explore; 1 = root only (default: no limit)"),
    :cg_log_level => (Int, "column generation log: 0 = off, 1 = one line per iteration (default: 0)"),
    :log_level => (Int, "branch-and-price log: 0 = off, 1 = one line per node, 2 = verbose (default: 1)"),
    :time_limit => (Float64, "wall-clock limit of the branch-and-price in seconds, for each node selection (default: 180)"),
    :seed => (Int, "seed of the random node selection (default: 1)"),
    :version => (:flag, "print the bgspprc version and exit"),
    :mono => (:flag, "forward labeling only (default: bidirectional)"),
    :stage => (String, "stage to start pricing in: heuristic1|heuristic2|exact (default: exact)"),
    :ng => (Int, "ng-neighbourhood size, at most 31; 0 disables (default: 0)"),
    :steps => (:steps, "bucket step sizes S1,S2 (default: 10,1)"),
    :xi => (Float64, "derive the steps from xi = X buckets per vertex (BG2021 6.3); not with --steps or --auto-steps"),
    :max_paths => (Int, "routes returned per pricing call; 0 = all (default: 1)"),
    :theta => (Float64, "pricing threshold θ of bgspprc: routes with reduced cost below it are kept (default: -1e-6)"),
    :edge_branching => (:flag, "branch on edge flows x_ij + x_ji instead of directed arcs"),
    :capacity_cuts => (:flag, "separate rounded capacity cuts"),
    :kpath_cuts => (:flag, "separate k-path cuts, k(S) = bin-packing number of S (robust form)"),
    :cut_rounds => (Int, "cut rounds per node (default: 20)"),
    :auto_steps => (:flag, "per-vertex automatic bucket steps"),
    :stats => (:flag, "print pricing statistics, summed over all calls"),
    :csv => (:flag, "machine-readable CSV output (header and one row)"),
    :timing => (:flag, "print the pricing phase timings, summed over all calls"),
    :no_parallel => (:flag, "sequential executor (default: parallel)"),
    :no_parallel_bidir => (:flag, "sequential forward/backward labeling"),
]

function usage()
    flag(name) = "--" * replace(String(name), '_' => '-')
    opts = join(("  $(rpad(flag(name) * (kind isa Type ? " <$(kind)>" : kind === :steps ? " S1,S2" : ""), 26)) $desc"
                 for (name, (kind, desc)) in CLI_OPTIONS), "\n")
    return "usage: julia --project=. cvrp.jl <instance.vrp> [options]\n\noptions (--name value, --name=value or name=value):\n$opts\n\n" *
        "example: julia --project=. cvrp.jl instances/A-n32-k5.vrp --ng 8 --max-paths 30 --capacity-cuts"
end

"""
    parse_args(args) -> (path, options::Dict{Symbol,Any})

The first non-option argument is the instance; options are `--name value`, `--name=value` or
`name=value` (dashes and underscores are interchangeable), and flags take no value.
"""
function parse_args(args)
    kinds = Dict(name => kind for (name, (kind, _)) in CLI_OPTIONS)
    path = nothing
    options = Dict{Symbol,Any}()
    i = 1
    while i <= length(args)
        arg = args[i]
        i += 1
        m = match(r"^(?:--?)?([A-Za-z][A-Za-z_-]*)=(.*)$", arg)
        m === nothing && (m = match(r"^--([A-Za-z][A-Za-z_-]*)$", arg))
        if m === nothing
            path === nothing || error("unexpected argument '$arg'\n\n$(usage())")
            path = arg
            continue
        end
        name = Symbol(replace(m[1], '-' => '_'))
        haskey(kinds, name) || error("unknown option '$(m[1])'\n\n$(usage())")
        kind = kinds[name]
        if kind === :flag
            length(m.captures) == 2 && m[2] !== nothing && error("option '$(m[1])' takes no value")
            options[name] = true
            continue
        end
        raw = if length(m.captures) == 2 && m[2] !== nothing
            m[2]
        else
            i <= length(args) || error("option '$(m[1])' needs a value")
            i += 1
            args[i-1]
        end
        value = if kind === :steps
            parts = tryparse.(Float64, split(raw, ','))
            length(parts) == 2 && all(!isnothing, parts) ? (parts[1], parts[2]) : nothing
        elseif kind === String
            raw
        else
            tryparse(kind, raw)
        end
        value === nothing && error("option '$(m[1])' expects $(kind === :steps ? "S1,S2" : kind), got '$raw'")
        options[name] = value
    end
    return path, options
end

# Pricing options of `build_cvrp_model` from the parsed command line.
function pricing_options(opts, inst::CVRPInstance)
    haskey(opts, :xi) && (haskey(opts, :steps) || haskey(opts, :auto_steps)) &&
        error("--xi derives the steps: drop --steps and --auto-steps")
    stage = Symbol(get(opts, :stage, "exact"))
    stage in BGSPPRCC.STAGES || error("--stage must be one of heuristic1|heuristic2|exact, got '$stage'")
    get(opts, :ng, 0) >= 0 || error("--ng must be >= 0")
    get(opts, :max_paths, 1) >= 0 || error("--max-paths must be >= 0")
    haskey(opts, :xi) && !(opts[:xi] > 0 && isfinite(opts[:xi])) && error("--xi must be positive and finite")
    steps = haskey(opts, :xi) ? (0.0, 0.0) : get(opts, :steps, (10.0, 1.0))   # steps <= 0 are derived from xi
    return (;
        ng=get(opts, :ng, 0), bucket_steps=steps, xi=get(opts, :xi, 25.0), max_paths=get(opts, :max_paths, 1),
        theta=get(opts, :theta, -1.0e-6), bidirectional=(!haskey(opts, :mono)), stage,
        auto_steps=haskey(opts, :auto_steps), parallel=(!haskey(opts, :no_parallel)),
        parallel_bidir=(!haskey(opts, :no_parallel) && !haskey(opts, :no_parallel_bidir)),
        edge_branching=haskey(opts, :edge_branching)
    )
end

const CSV_HEADER = "name,type,n_verts,n_arcs,theta,cost,n_paths,ms,n_buckets,n_labels_created," *
    "n_dominance_checks,n_non_dominated,n_fixed_buckets,n_eliminated_arcs,label_state_bytes," *
    "n_dominance_checks_fw,n_dominance_checks_bw,n_non_dominated_fw,n_non_dominated_bw,strategy"

function print_report(opts, inst, out, totals, theta, elapsed, strategy; header=true)
    t(k) = get(totals, k, 0.0)
    i(k) = round(Int, t(k))
    root = out.root_lp_value
    if haskey(opts, :csv)
        header && println(CSV_HEADER)
        println(join([
            inst.name, "cvrp", i(:n_verts), i(:n_arcs), theta, root, i(:paths), round(1000elapsed; digits=1),
            i(:n_buckets), i(:labels_created), i(:dominance_checks), i(:non_dominated_labels),
            i(:n_fixed_buckets), i(:eliminated_bucket_arcs), i(:label_state_bytes),
            i(:dominance_checks_fw), i(:dominance_checks_bw), i(:non_dominated_fw), i(:non_dominated_bw),
            strategy,
        ], ","))
        return
    end
    println("$(inst.name) [$strategy]: status=$(out.status) nodes=$(out.nodes_explored) root_lp=$root " *
        "primal=$(out.incumbent === nothing ? "-" : out.incumbent.obj_value) time=$(round(elapsed; digits = 2))s")
    if haskey(totals, :cut_rounds)
        println("  cuts: capacity=$(i(:capacity_cuts))  kpath=$(i(:kpath_cuts))  separation calls=$(i(:cut_rounds))")
    end
    if haskey(opts, :stats)
        println("  pricing calls=$(i(:calls))  paths=$(i(:paths))")
        println("  n_buckets=$(i(:n_buckets))  n_labels_created=$(i(:labels_created))  " *
            "n_dominance_checks=$(i(:dominance_checks))  n_non_dominated=$(i(:non_dominated_labels))")
        println("  n_dominance_checks_fw=$(i(:dominance_checks_fw))  n_dominance_checks_bw=$(i(:dominance_checks_bw))  " *
            "n_non_dominated_fw=$(i(:non_dominated_fw))  n_non_dominated_bw=$(i(:non_dominated_bw))")
        println("  n_fixed_buckets=$(i(:n_fixed_buckets))  n_eliminated_arcs=$(i(:eliminated_bucket_arcs))  " *
            "label_state_bytes=$(i(:label_state_bytes))")
    end
    if haskey(opts, :timing)
        fw, bw, cb, cc, pe = t(:forward_labeling_ms), t(:backward_labeling_ms), t(:completion_bounds_ms),
        t(:concatenation_ms), t(:path_extraction_ms)
        println("  timing:  fw=$(round(fw; digits = 3))ms  bw=$(round(bw; digits = 3))ms  " *
            "completion=$(round(cb; digits = 3))ms  concat=$(round(cc; digits = 3))ms  " *
            "paths=$(round(pe; digits = 3))ms  sum=$(round(fw + bw + cb + cc + pe; digits = 3))ms")
    end
end

function (@main)(args)
    if any(in(("-h", "--help", "help")), args)
        println(usage())
        return
    end
    path, opts = parse_args(args)
    if haskey(opts, :version)
        println("bgspprc ", BGSPPRCC.version())
        return
    end
    path === nothing && error("missing instance file\n\n$(usage())")
    QUIET[] = haskey(opts, :csv)
    inst = parse_cvrp(path; customers=get(opts, :customers, nothing), fleet=get(opts, :fleet, nothing))
    popts = pricing_options(opts, inst)
    progress("$(inst.name): $(n_customers(inst)) customers, capacity $(inst.capacity), fleet $(inst.fleet)")
    # The same parameters for every node selection; only the strategy changes.
    results = []
    for (k, (strategy, node_select_strategy)) in enumerate(node_strategies(get(opts, :seed, 1)))
        progress("node selection: $strategy")
        elapsed = @elapsed (out, totals) = solve_cvrp(
            inst; node_limit=get(opts, :node_limit, typemax(Int)),
            time_limit=get(opts, :time_limit, 180.0), silent=haskey(opts, :csv), node_select_strategy,
            cg_log_level=get(opts, :cg_log_level, 0), log_level=get(opts, :log_level, 1),
            capacity_cuts=haskey(opts, :capacity_cuts), kpath_cuts=haskey(opts, :kpath_cuts),
            cut_rounds=get(opts, :cut_rounds, 20), popts...
        )
        print_report(opts, inst, out, totals, popts.theta, elapsed, strategy; header=(k == 1))
        push!(results, (strategy, out, elapsed))
    end
    if !haskey(opts, :csv)
        println("\n", rpad("strategy", 13), rpad("status", 12), rpad("nodes", 8), rpad("primal", 12), "time")
        for (strategy, out, elapsed) in results
            primal = out.incumbent === nothing ? "-" : string(round(out.incumbent.obj_value; digits=2))
            println(rpad(strategy, 13), rpad(out.status, 12), rpad(out.nodes_explored, 8), rpad(primal, 12),
                round(elapsed; digits=1), "s")
        end
    end
end
