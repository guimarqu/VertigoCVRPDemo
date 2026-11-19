# Independent reference: enumerate every capacity-feasible elementary route
# (Held-Karp per customer subset), then solve the set-partitioning LP and IP
# with HiGHS. Only for small instances (2^customers subsets).
#
#   julia --project=. tools/reference.jl instances/A-n32-k5.vrp <customers> <fleet>
using JuMP, HiGHS
module Demo   # in a module, so that the @main of cvrp.jl is not the entry point
include("../cvrp.jl")
end
using .Demo: parse_cvrp, n_customers
function reference(path, customers, fleet)
    inst = parse_cvrp(path; customers, fleet)
    n = n_customers(inst); d = inst.dist; q = inst.demand
    # Held-Karp: best[S][last] = min cost depot -> ... -> last visiting S
    N = 1 << n
    best = fill(Inf, N, n)
    for i in 1:n; best[1 << (i-1) + 1, i] = d[1, i+1]; end
    for S in 1:(N-1), i in 1:n
        isinf(best[S+1, i]) && continue
        for j in 1:n
            (S >> (j-1)) & 1 == 1 && continue
            T = S | (1 << (j-1))
            best[T+1, j] = min(best[T+1, j], best[S+1, i] + d[i+1, j+1])
        end
    end
    routes = Tuple{Vector{Int},Float64}[]
    for S in 1:(N-1)
        members = [i for i in 1:n if (S >> (i-1)) & 1 == 1]
        sum(q[i+1] for i in members) <= inst.capacity || continue
        push!(routes, (members, minimum(best[S+1, i] + d[i+1, 1] for i in members)))
    end
    res = map((false, true)) do integer
        m = Model(HiGHS.Optimizer); set_silent(m)
        @variable(m, λ[eachindex(routes)] >= 0, integer = integer)
        @constraint(m, [i in 1:n], sum(λ[r] for r in eachindex(routes) if i in routes[r][1]) == 1)
        @constraint(m, sum(λ) <= inst.fleet)
        @objective(m, Min, sum(routes[r][2] * λ[r] for r in eachindex(routes)))
        optimize!(m)
        termination_status(m) == OPTIMAL ? objective_value(m) : termination_status(m)
    end
    println("customers=$customers fleet=$(inst.fleet) total_demand=$(sum(q[2:end])) routes=$(length(routes)) LP=$(res[1]) IP=$(res[2])")
end
reference(ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]))
