#pragma once
// mmgen: generates a GEMM from a nest, and runs it.
//
//     #include <mmgen/gemm.hpp>
//
//     mmgen::problem p{2000, 2000, 2000};
//     mmgen::aligned_allocator alloc;                       // or your own
//     mmgen::gemm g(provider, "n:768 k:768 | m:120 k:192 | n:nr m:mr", p, alloc);
//     g(alpha, A, lda, B, ldb, beta, C, rs_c, cs_c);        // as often as needed
//
// The provider is the leaf: BLIS's kernel, a ukrgen kernel, or a reference
// loop (see ukr_provider.hpp).  The plan is the "gen": a nest of the MOMMS
// family (Smith and van de Geijn, SC19), its packers, and that leaf, built once
// for a problem and then used as a GEMM.  There is no BLAS surface here -- no
// transposes, no packing of C, no threading: one GEMM, exactly as the nest says
// it, on the layouts the caller gives.
//
// A plan carries its own benchmarking state.  With `profile` set, each run()
// records the micro-kernel calls, the packing (calls, elements, time) and the
// per-rung iteration counts, so a benchmark can read what happened instead of
// guessing: see `stats()`.

#include <string>

#include "mmgen/allocator.hpp"
#include "mmgen/nest.hpp"
#include "mmgen/packer.hpp"
#include "mmgen/rung.hpp"
#include "mmgen/types.hpp"
#include "mmgen/ukr_provider.hpp"

namespace mmgen {

/// what to generate, and how to watch it
struct gemm_config
{
    std::string pack_at;    // "A:LABEL,B:LABEL" or "pre"; empty: the model's default
    int top = 0;            // outermost rung that iterates; 0: the whole nest
    bool profile = false;   // record calls, packing and per-rung iterations
};

/// A GEMM generated from a nest.
class gemm
{
public:
    gemm(ukr_provider& provider, const std::string& nest_text, const problem& p,
         buffer_allocator& alloc, const gemm_config& cfg = {})
        : gemm(provider, parse_nest(nest_text, provider.info().mr, provider.info().nr,
                                    cfg.pack_at), p, alloc, cfg)
    {
    }

    gemm(ukr_provider& provider, const nest_spec& spec, const problem& p,
         buffer_allocator& alloc, const gemm_config& cfg = {})
        : nest_(provider, spec, p, alloc)
    {
        nest_.set_top(cfg.top);
        nest_.set_profiling(cfg.profile);
    }

    gemm(const gemm&) = delete;
    gemm& operator=(const gemm&) = delete;

    /// C = alpha*A*B + beta*C, as the nest walks it.
    void operator()(double alpha, const void* a, stride_t lda, const void* b, stride_t ldb,
                    double beta, void* c, stride_t rs_c, stride_t cs_c)
    {
        call k;
        k.a = a;
        k.lda = lda;
        k.b = b;
        k.ldb = ldb;
        k.c = c;
        k.rs_c = rs_c;
        k.cs_c = cs_c;
        k.alpha = alpha;
        k.beta = beta;
        nest_.run(k);
    }

    void operator()(const call& k) { nest_.run(k); }

    /// pack ahead of a call whose matrices do not change (run() does this too)
    void prepare(const call& k) { nest_.prepack(k); }

    // ---- introspection, for a benchmark or the model's cross-check ----
    const nest& plan() const { return nest_; }
    const kernel_info& kernel() const { return nest_.kernel(); }
    const profile& stats() const { return nest_.stats(); }
    void reset_stats() { nest_.stats().reset(); }
    const problem& size() const { return nest_.size(); }
    std::string text() const { return nest_.text(); }
    std::string loops() const { return nest_.loops_text(); }
    std::string packing() const { return nest_.pack_text(); }
    std::size_t pack_bytes() const { return nest_.pack_bytes(); }
    std::size_t block_elems(operand op) const { return nest_.block_elems(op); }
    std::size_t slots(operand op) const { return nest_.slots(op); }
    double calls() const { return nest_.calls(); }
    double work() const { return nest_.work(); }
    index_t mr() const { return nest_.mr(); }
    index_t nr() const { return nest_.nr(); }

private:
    nest nest_;
};

} // namespace mmgen
