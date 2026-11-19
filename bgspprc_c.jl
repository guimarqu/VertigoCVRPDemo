# Thin ccall wrapper over libbgspprc_c, the C shim in capi/bgspprc_c.cpp over the
# header-only bucket-graph-spprc library.
#
# Build the library first with deps/build.sh, or point BGSPPRC_C_LIB at another build.

module BGSPPRCC

const LIB = get(
    ENV, "BGSPPRC_C_LIB",
    joinpath(@__DIR__, "deps", "build", "libbgspprc_c." * Base.Libc.Libdl.dlext)
)

# Mirrors `bgs_options`.
struct Options
    bucket_step_1::Cdouble
    bucket_step_2::Cdouble
    xi::Cdouble
    max_paths::Int32
    bidirectional::Int32
    stage::Int32
    auto_steps::Int32
    parallel::Int32
    parallel_bidir::Int32
end

# Stage names of `bgs_stage`, in order.
const STAGES = (:heuristic1, :heuristic2, :exact)

version() = unsafe_string(@ccall LIB.bgs_version()::Cstring)

lasterror() = unsafe_string(@ccall LIB.bgs_last_error()::Cstring)

check(status::Cint, what) = status == 0 || error("$what: $(lasterror())")

mutable struct Solver
    ptr::Ptr{Cvoid}
end

"""
    Solver(; n_vertices, source, sink, arc_from, arc_to, arc_cost, arc_resource,
             vertex_lb, vertex_ub, n_main = 1, ng = nothing, bucket_steps = (10.0, 1.0),
             xi = 25.0, max_paths = 1, bidirectional = true, stage = :exact, auto_steps = false,
             parallel = true, parallel_bidir = true)

Build a bucket-graph solver. Vertices are 1-based; `arc_resource[r][a]` and
`vertex_lb[r][v]`, `vertex_ub[r][v]` give resource `r`, the first `n_main` of which
span the bucket grid. `ng[v]` lists the ng-neighbours of `v` (at most 31). Without `ng`,
elementarity is not enforced. A bucket step `<= 0` is derived from `xi`, the maximum number of buckets
per vertex (BG2021 §6.3): `range / xi`, rounded to 0.1. The keyword defaults are those of `bgspprc-solve`; `stage` is the
stage the solver starts in (`:heuristic1`, `:heuristic2` or `:exact`), after which it progresses by itself.
"""
function Solver(;
        n_vertices, source, sink, arc_from, arc_to, arc_cost, arc_resource, vertex_lb, vertex_ub,
        n_main = 1, ng = nothing, bucket_steps = (10.0, 1.0), xi = 25.0, max_paths = 1, bidirectional = true,
        stage = :exact, auto_steps = false, parallel = true, parallel_bidir = true
    )
    n_arcs, n_res = length(arc_from), length(arc_resource)
    from = Int32.(arc_from .- 1)
    to = Int32.(arc_to .- 1)
    cost = Float64.(arc_cost)
    ares = Float64.(reduce(vcat, arc_resource))
    lb = Float64.(reduce(vcat, vertex_lb))
    ub = Float64.(reduce(vcat, vertex_ub))
    ng_off, ng_nbr = if ng === nothing
        nothing, nothing
    else
        Int32.(vcat(0, cumsum(length.(ng)))), Int32.(reduce(vcat, ng) .- 1)
    end
    stage_id = findfirst(==(Symbol(stage)), STAGES)
    stage_id === nothing && error("stage must be one of $STAGES, got $stage")
    opts = Options(
        bucket_steps[1], bucket_steps[2], xi, max_paths, bidirectional, stage_id - 1, auto_steps, parallel, parallel_bidir
    )
    ptr = GC.@preserve from to cost ares lb ub ng_off ng_nbr begin
        @ccall LIB.bgs_create(
            n_vertices::Int32, (source - 1)::Int32, (sink - 1)::Int32, n_arcs::Int32,
            from::Ptr{Int32}, to::Ptr{Int32}, cost::Ptr{Cdouble},
            n_res::Int32, n_main::Int32, ares::Ptr{Cdouble}, lb::Ptr{Cdouble}, ub::Ptr{Cdouble},
            (ng_off === nothing ? C_NULL : pointer(ng_off))::Ptr{Int32},
            (ng_nbr === nothing ? C_NULL : pointer(ng_nbr))::Ptr{Int32},
            Ref(opts)::Ref{Options}
        )::Ptr{Cvoid}
    end
    ptr == C_NULL && error("bgs_create: $(lasterror())")
    return finalizer(destroy!, Solver(ptr))
end

function destroy!(s::Solver)
    if s.ptr != C_NULL
        @ccall LIB.bgs_destroy(s.ptr::Ptr{Cvoid})::Cvoid
        s.ptr = C_NULL
    end
    return nothing
end

"""
    solve!(s, arc_costs, θ) -> (paths, exact)

Paths with reduced cost below `θ`, each a NamedTuple with 1-based `vertices` and `arcs`
and its `reduced_cost` and `original_cost`. `arc_costs` replaces the costs used from now on
(`nothing` keeps the previous ones). `exact` tells whether the exact stage produced the result:
only then does an empty result prove that no path is below `θ`, and does the best path
equal the optimum.
"""
function solve!(s::Solver, arc_costs::Union{Nothing,Vector{Float64}}, θ)
    n, exact = Ref{Int32}(0), Ref{Int32}(0)
    GC.@preserve arc_costs check(
        @ccall(
            LIB.bgs_solve(
                s.ptr::Ptr{Cvoid}, (arc_costs === nothing ? C_NULL : pointer(arc_costs))::Ptr{Cdouble},
                θ::Cdouble, n::Ref{Int32}, exact::Ref{Int32}
            )::Cint
        ), "bgs_solve"
    )
    paths = map(0:(n[] - 1)) do i
        rc, oc, nv, na = Ref(0.0), Ref(0.0), Ref{Int32}(0), Ref{Int32}(0)
        check(
            @ccall(LIB.bgs_path_info(s.ptr::Ptr{Cvoid}, i::Int32, rc::Ref{Cdouble}, oc::Ref{Cdouble}, nv::Ref{Int32}, na::Ref{Int32})::Cint),
            "bgs_path_info"
        )
        vertices, arcs = Vector{Int32}(undef, nv[]), Vector{Int32}(undef, na[])
        check(
            @ccall(LIB.bgs_path_copy(s.ptr::Ptr{Cvoid}, i::Int32, vertices::Ptr{Int32}, arcs::Ptr{Int32})::Cint),
            "bgs_path_copy"
        )
        (; vertices = Int.(vertices) .+ 1, arcs = Int.(arcs) .+ 1, reduced_cost = rc[], original_cost = oc[])
    end
    return paths, exact[] != 0
end

# Mirrors `bgs_stats`.
struct Stats
    labels_created::Int64
    dominance_checks::Int64
    dominance_checks_fw::Int64
    dominance_checks_bw::Int64
    non_dominated_labels::Int64
    non_dominated_fw::Int64
    non_dominated_bw::Int64
    eliminated_bucket_arcs::Int64
    label_state_bytes::Int64
    n_buckets::Int32
    n_fixed_buckets::Int32
    forward_labeling_ms::Cdouble
    backward_labeling_ms::Cdouble
    completion_bounds_ms::Cdouble
    concatenation_ms::Cdouble
    path_extraction_ms::Cdouble
end

"Statistics of the last `solve!`, summed over the stage retries it made."
function stats(s::Solver)
    st = Ref{Stats}()
    check(@ccall(LIB.bgs_get_stats(s.ptr::Ptr{Cvoid}, st::Ref{Stats})::Cint), "bgs_get_stats")
    return st[]
end

end # module
