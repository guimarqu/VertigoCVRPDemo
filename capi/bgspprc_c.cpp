// C shim over the header-only bucket-graph-spprc library (upstream has no C API),
// written to match the
// calls of bgspprc_c.jl (bgs_create, bgs_solve, bgs_path_info, bgs_path_copy,
// bgs_get_stats, bgs_version, bgs_last_error, bgs_destroy).
//
// Mirrors cli/main.cpp: Solver<NgPack|EmptyPack, StdThread|Sequential>,
// optional per-vertex auto steps, build(), set_stage(start stage), solve().
// The threshold θ is a Solver option fixed at construction, while bgs_solve
// takes it per call: the solver is built on the first bgs_solve, and a later
// call with another θ is an error.

#include <bgspprc/bgspprc.h>
#include <bgspprc/executor_thread.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace bgspprc;

extern "C" {

struct bgs_options {
    double bucket_step_1;
    double bucket_step_2;
    double xi;
    int32_t max_paths;
    int32_t bidirectional;
    int32_t stage;  // 0 heuristic1, 1 heuristic2, 2 exact
    int32_t auto_steps;
    int32_t parallel;
    int32_t parallel_bidir;
};

struct bgs_stats {
    int64_t labels_created;
    int64_t dominance_checks;
    int64_t dominance_checks_fw;
    int64_t dominance_checks_bw;
    int64_t non_dominated_labels;
    int64_t non_dominated_fw;
    int64_t non_dominated_bw;
    int64_t eliminated_bucket_arcs;
    int64_t label_state_bytes;
    int32_t n_buckets;
    int32_t n_fixed_buckets;
    double forward_labeling_ms;
    double backward_labeling_ms;
    double completion_bounds_ms;
    double concatenation_ms;
    double path_extraction_ms;
};

}  // extern "C"

namespace {

thread_local std::string g_last_error;

struct PathOut {
    std::vector<int> vertices;
    std::vector<int> arcs;
    double reduced_cost;
    double original_cost;
};

// Type-erased solver: one implementation per (pack, executor) pair.
struct AnySolver {
    virtual ~AnySolver() = default;
    virtual void solve(std::vector<PathOut>& out, bool& exact) = 0;
    virtual void update_costs(const double* costs, int n) = 0;
    virtual bgs_stats stats() const = 0;
};

template <typename Pack, typename Exec>
struct SolverImpl final : AnySolver {
    Solver<Pack, Exec> solver;

    SolverImpl(const ProblemView& pv, Pack pack, typename Solver<Pack, Exec>::Options so,
               bool auto_steps, Stage stage, const double* costs)
        : solver(pv, std::move(pack), so, Exec{}) {
        if (auto_steps) {
            solver.set_vertex_bucket_steps(solver.compute_min_inbound_arc_resource());
        }
        solver.build();
        solver.set_stage(stage);
        solver.update_arc_costs(std::span<const double>(costs, pv.n_arcs));
    }

    // Marks the costs dirty: the bucket graph only re-reads them then.
    void update_costs(const double* costs, int n) override {
        solver.update_arc_costs(std::span<const double>(costs, n));
    }

    void solve(std::vector<PathOut>& out, bool& exact) override {
        auto paths = solver.solve();
        out.clear();
        out.reserve(paths.size());
        for (auto& p : paths) {
            out.push_back({std::move(p.vertices), std::move(p.arcs), p.reduced_cost,
                           p.original_cost});
        }
        exact = solver.exact_complete();
    }

    bgs_stats stats() const override {
        const auto& t = solver.solve_timings();
        return {
            .labels_created = solver.labels_created(),
            .dominance_checks = solver.dominance_checks(),
            .dominance_checks_fw = solver.dominance_checks(Direction::Forward),
            .dominance_checks_bw = solver.dominance_checks(Direction::Backward),
            .non_dominated_labels = solver.non_dominated_labels(),
            .non_dominated_fw = solver.non_dominated_labels(Direction::Forward),
            .non_dominated_bw = solver.non_dominated_labels(Direction::Backward),
            .eliminated_bucket_arcs = solver.eliminated_bucket_arcs(),
            .label_state_bytes = static_cast<int64_t>(Solver<Pack, Exec>::label_state_size()),
            .n_buckets = solver.n_buckets(),
            .n_fixed_buckets = solver.n_fixed_buckets(),
            .forward_labeling_ms = t.forward_labeling.count(),
            .backward_labeling_ms = t.backward_labeling.count(),
            .completion_bounds_ms = t.completion_bounds.count(),
            .concatenation_ms = t.concatenation.count(),
            .path_extraction_ms = t.path_extraction.count(),
        };
    }
};

using NgPack = ResourcePack<NgPathResource>;

struct Handle {
    // Owned copies of the problem data: ProblemView only borrows.
    std::vector<int> from, to;
    std::vector<double> base_cost, costs;
    std::vector<std::vector<double>> ares, lb, ub;
    std::vector<const double*> ares_p, lb_p, ub_p;
    std::vector<std::vector<int>> ng;  // empty: no ng-path resource
    ProblemView pv;
    bgs_options opts;

    std::unique_ptr<AnySolver> solver;  // built on the first solve
    double theta = 0.0;
    std::vector<PathOut> paths;

    template <typename Exec>
    std::unique_ptr<AnySolver> make(double th) {
        Stage stage = static_cast<Stage>(opts.stage);
        auto so_for = [&]<typename Pack>() {
            typename Solver<Pack, Exec>::Options so{
                .bucket_steps = {opts.bucket_step_1, opts.bucket_step_2},
                .xi = opts.xi,
                .bidirectional = opts.bidirectional != 0,
                .parallel_bidir = opts.parallel != 0 && opts.parallel_bidir != 0,
                .max_paths = opts.max_paths,
                .theta = th,
            };
            return so;
        };
        if (!ng.empty()) {
            NgPathResource res(pv.n_vertices, pv.n_arcs, pv.arc_from, pv.arc_to, ng);
            return std::make_unique<SolverImpl<NgPack, Exec>>(
                pv, make_resource_pack(std::move(res)), so_for.template operator()<NgPack>(),
                opts.auto_steps != 0, stage, costs.data());
        }
        return std::make_unique<SolverImpl<EmptyPack, Exec>>(
            pv, EmptyPack{}, so_for.template operator()<EmptyPack>(), opts.auto_steps != 0,
            stage, costs.data());
    }
};

template <typename F>
int guarded(F&& f) {
    try {
        f();
        return 0;
    } catch (const std::exception& e) {
        g_last_error = e.what();
    } catch (...) {
        g_last_error = "unknown C++ exception";
    }
    return 1;
}

}  // namespace

extern "C" {

const char* bgs_version() { return "0.2.0-capi-shim"; }

const char* bgs_last_error() { return g_last_error.c_str(); }

void* bgs_create(int32_t n_vertices, int32_t source, int32_t sink, int32_t n_arcs,
                 const int32_t* arc_from, const int32_t* arc_to, const double* arc_cost,
                 int32_t n_res, int32_t n_main, const double* ares, const double* vlb,
                 const double* vub, const int32_t* ng_off, const int32_t* ng_nbr,
                 const bgs_options* opts) {
    Handle* h = nullptr;
    int rc = guarded([&] {
        auto hp = std::make_unique<Handle>();
        hp->from.assign(arc_from, arc_from + n_arcs);
        hp->to.assign(arc_to, arc_to + n_arcs);
        hp->base_cost.assign(arc_cost, arc_cost + n_arcs);
        hp->costs = hp->base_cost;
        for (int r = 0; r < n_res; ++r) {
            hp->ares.emplace_back(ares + static_cast<size_t>(r) * n_arcs,
                                  ares + static_cast<size_t>(r + 1) * n_arcs);
            hp->lb.emplace_back(vlb + static_cast<size_t>(r) * n_vertices,
                                vlb + static_cast<size_t>(r + 1) * n_vertices);
            hp->ub.emplace_back(vub + static_cast<size_t>(r) * n_vertices,
                                vub + static_cast<size_t>(r + 1) * n_vertices);
        }
        for (int r = 0; r < n_res; ++r) {
            hp->ares_p.push_back(hp->ares[r].data());
            hp->lb_p.push_back(hp->lb[r].data());
            hp->ub_p.push_back(hp->ub[r].data());
        }
        if (ng_off != nullptr && ng_nbr != nullptr) {
            hp->ng.resize(n_vertices);
            for (int v = 0; v < n_vertices; ++v) {
                hp->ng[v].assign(ng_nbr + ng_off[v], ng_nbr + ng_off[v + 1]);
            }
        }
        hp->pv = ProblemView{
            .n_vertices = n_vertices,
            .source = source,
            .sink = sink,
            .n_arcs = n_arcs,
            .arc_from = hp->from.data(),
            .arc_to = hp->to.data(),
            .arc_base_cost = hp->base_cost.data(),
            .n_resources = n_res,
            .arc_resource = hp->ares_p.data(),
            .vertex_lb = hp->lb_p.data(),
            .vertex_ub = hp->ub_p.data(),
            .n_main_resources = n_main,
        };
        hp->opts = *opts;
        h = hp.release();
    });
    return rc == 0 ? h : nullptr;
}

void bgs_destroy(void* s) { delete static_cast<Handle*>(s); }

int bgs_solve(void* s, const double* arc_costs, double theta, int32_t* n_paths, int32_t* exact) {
    auto* h = static_cast<Handle*>(s);
    return guarded([&] {
        if (arc_costs != nullptr) {
            // The solver keeps a pointer to this buffer: copy in place.
            std::copy(arc_costs, arc_costs + h->pv.n_arcs, h->costs.begin());
        }
        if (!h->solver) {
            h->theta = theta;
            h->solver = h->opts.parallel != 0 ? h->make<StdThreadExecutor>(theta)
                                              : h->make<SequentialExecutor>(theta);
        } else if (theta != h->theta) {
            throw std::invalid_argument("theta cannot change after the first solve");
        } else if (arc_costs != nullptr) {
            h->solver->update_costs(h->costs.data(), h->pv.n_arcs);
        }
        bool ex = false;
        h->solver->solve(h->paths, ex);
        *n_paths = static_cast<int32_t>(h->paths.size());
        *exact = ex ? 1 : 0;
    });
}

int bgs_path_info(void* s, int32_t i, double* rc, double* oc, int32_t* nv, int32_t* na) {
    auto* h = static_cast<Handle*>(s);
    return guarded([&] {
        const auto& p = h->paths.at(static_cast<size_t>(i));
        *rc = p.reduced_cost;
        *oc = p.original_cost;
        *nv = static_cast<int32_t>(p.vertices.size());
        *na = static_cast<int32_t>(p.arcs.size());
    });
}

int bgs_path_copy(void* s, int32_t i, int32_t* vertices, int32_t* arcs) {
    auto* h = static_cast<Handle*>(s);
    return guarded([&] {
        const auto& p = h->paths.at(static_cast<size_t>(i));
        std::copy(p.vertices.begin(), p.vertices.end(), vertices);
        std::copy(p.arcs.begin(), p.arcs.end(), arcs);
    });
}

int bgs_get_stats(void* s, bgs_stats* out) {
    auto* h = static_cast<Handle*>(s);
    return guarded([&] {
        if (!h->solver) {
            throw std::logic_error("no solve yet");
        }
        *out = h->solver->stats();
    });
}

}  // extern "C"
