# Robust CVRP cuts for Vertigo's CustomCutSeparator.
#
# Both families read  x(δ(S)) >= 2 k(S)  for a customer set S, on the edge flow
# of the master solution projected on the arc variables (the sink is the depot):
#   - rounded capacity: k(S) = ceil(q(S) / Q);
#   - k-path (robust form): k(S) = the bin-packing number of the demands of S,
#     never below the rounded capacity value. The strong k-path and route-load
#     knapsack cuts of BALDES are non-robust (their coefficient depends on the
#     whole route), which Vertigo's robust separator cannot express.
#
# Every customer has inflow 1 and outflow 1, so x(δ(S)) = 2|S| - 2 x(E(S)).

# Edge {u, v} (u < v, depot = 1) of every original arc variable, and the original
# variables of each edge. With --edge-branching both arcs of an edge share one.
struct CVRPCutGraph
    demand::Vector{Float64}
    capacity::Float64
    customers::UnitRange{Int}
    edge_of::Dict{JuMP.VariableRef,Tuple{Int,Int}}
    vars_of_edge::Dict{Tuple{Int,Int},Vector{JuMP.VariableRef}}
end

function CVRPCutGraph(inst, arcs, decomp, sp_id, arc_vars, sink)
    node(v) = v == sink ? 1 : v
    edge_of = Dict{JuMP.VariableRef,Tuple{Int,Int}}()
    vars_of_edge = Dict{Tuple{Int,Int},Vector{JuMP.VariableRef}}()
    for (a, sp_var) in zip(arcs, arc_vars)
        ov = Vertigo.Reformulation.mapped_original_var(decomp, sp_id, sp_var)
        u, v = minmax(node(a[1]), node(a[2]))
        edge_of[ov] = (u, v)
        vars = get!(vars_of_edge, (u, v), JuMP.VariableRef[])
        ov in vars || push!(vars, ov)
    end
    return CVRPCutGraph(inst.demand, inst.capacity, 2:(n_customers(inst)+1), edge_of, vars_of_edge)
end

"""
    _fits_in_bins(sizes, capacity, k; node_limit) -> Union{Bool,Nothing}

Whether `sizes` (sorted decreasingly) pack into `k` bins of `capacity`, by depth-first search;
`nothing` when `node_limit` nodes did not settle it.
"""
function _fits_in_bins(sizes, capacity, k; node_limit=100_000)
    loads = zeros(k)
    nodes = Ref(0)
    function place(i)
        i > length(sizes) && return true
        nodes[] += 1
        nodes[] > node_limit && return nothing
        tried = Float64[]
        for b in 1:k
            # Bins with the same load are interchangeable: try one of them.
            if loads[b] + sizes[i] <= capacity + 1e-9 && !(loads[b] in tried)
                push!(tried, loads[b])
                loads[b] += sizes[i]
                r = place(i + 1)
                loads[b] -= sizes[i]
                r === true && return true
                r === nothing && return nothing
            end
        end
        return false
    end
    return place(1)
end

"""
    bin_packing_bound(sizes, capacity, k0) -> Int

The bin-packing number of `sizes`, or the smallest `k >= k0` not proved infeasible
within the search budget: a valid lower bound either way.
"""
function bin_packing_bound(sizes, capacity, k0)
    sorted = sort(sizes; rev=true)
    k = k0
    while _fits_in_bins(sorted, capacity, k) === false
        k += 1
    end
    return k
end

# Customer sets to test: the connected components of the support graph, and
# the sets grown greedily from each customer along the heaviest edge.
function _candidate_sets(g::CVRPCutGraph, y)
    w = Dict{Int,Dict{Int,Float64}}(i => Dict{Int,Float64}() for i in g.customers)
    for ((u, v), val) in y
        if u != 1 && val > 1e-6
            w[u][v] = get(w[u], v, 0.0) + val
            w[v][u] = get(w[v], u, 0.0) + val
        end
    end
    sets = Set{Vector{Int}}()
    seen = Set{Int}()
    for i in g.customers
        i in seen && continue
        comp, stack = Int[], [i]
        push!(seen, i)
        while !isempty(stack)
            u = pop!(stack)
            push!(comp, u)
            for v in keys(w[u])
                if !(v in seen)
                    push!(seen, v)
                    push!(stack, v)
                end
            end
        end
        push!(sets, sort(comp))
    end
    for i in g.customers
        S = [i]
        conn = copy(w[i])
        while length(S) < length(g.customers)
            best, j = -1.0, 0
            for (v, c) in conn
                if !(v in S) && (c > best || (c == best && v < j))
                    best, j = c, v
                end
            end
            j == 0 && break
            push!(S, j)
            for (v, c) in w[j]
                conn[v] = get(conn, v, 0.0) + c
            end
            push!(sets, sort(S))
        end
    end
    return sets
end

mutable struct CVRPCutSeparator <: Vertigo.AbstractCutSeparator
    graph::CVRPCutGraph
    capacity_cuts::Bool
    kpath_cuts::Bool
    max_cuts::Int                      # per separation round
    added::Set{Vector{Int}}
    counts::Dict{Symbol,Int}
end

CVRPCutSeparator(graph; capacity_cuts, kpath_cuts, max_cuts=50) =
    CVRPCutSeparator(graph, capacity_cuts, kpath_cuts, max_cuts, Set{Vector{Int}}(),
        Dict(:capacity => 0, :kpath => 0, :rounds => 0))

function Vertigo.separate(sep::CVRPCutSeparator, x)
    g = sep.graph
    sep.counts[:rounds] += 1
    y = Dict{Tuple{Int,Int},Float64}()
    for (ov, val) in x
        e = get(g.edge_of, ov, nothing)
        if e !== nothing
            y[e] = get(y, e, 0.0) + val
        end
    end
    found = Tuple{Float64,Vector{Int},Int,Symbol}[]
    for S in _candidate_sets(g, y)
        S in sep.added && continue
        inS = Set(S)
        inside = sum((val for ((u, v), val) in y if u in inS && v in inS); init=0.0)
        cross = 2 * length(S) - 2 * inside
        q = sum(g.demand[i] for i in S)
        k_rcc = ceil(Int, q / g.capacity - 1e-9)
        k, family = k_rcc, :capacity
        if sep.kpath_cuts
            k_bpp = bin_packing_bound([g.demand[i] for i in S], g.capacity, k_rcc)
            if k_bpp > k_rcc
                k, family = k_bpp, :kpath
            end
        end
        if family == :capacity && !sep.capacity_cuts
            continue
        end
        violation = 2k - cross
        if violation > 1e-4
            push!(found, (violation, S, k, family))
        end
    end
    sort!(found; by=first, rev=true)
    cuts = Vertigo.SeparatedCut{JuMP.VariableRef}[]
    for (_, S, k, family) in first(found, sep.max_cuts)
        push!(sep.added, S)
        sep.counts[family] += 1
        inS = Set(S)
        coeffs = Dict{JuMP.VariableRef,Float64}()
        for ((u, v), vars) in g.vars_of_edge
            if (u in inS) != (v in inS)
                for ov in vars
                    coeffs[ov] = 1.0
                end
            end
        end
        push!(cuts, Vertigo.SeparatedCut{JuMP.VariableRef}(coeffs, MOI.GreaterThan(2.0 * k)))
    end
    return cuts
end
