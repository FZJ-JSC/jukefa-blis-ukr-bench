#pragma once
// BLIS's micro-kernel and pack kernel, adapted to the momms harness's two
// function-pointer types: a ukernel and a packm.
//
// Nothing here is required by a nest -- that is the point.  A ukrgen kernel
// brings its own gemm_fn (and, for a datatype BLIS has no kernel for, its own
// packm_fn) and skips this header entirely.

#include <string>

#include "blis_kernels.hpp"
#include "nest.hpp"

namespace momms {

struct blis_gemm_ctx
{
    const gotobench::blis_kernels* ker = nullptr;
};

inline void blis_gemm(void* ctx, dim_t m, dim_t n, dim_t k, const void* alpha,
                      const void* a, const void* b, const void* beta, void* c,
                      inc_t rs_c, inc_t cs_c, auxinfo_t* aux)
{
    auto* g = static_cast<blis_gemm_ctx*>(ctx);
    g->ker->gemm(m, n, k, static_cast<const double*>(alpha),
                 static_cast<const double*>(a), static_cast<const double*>(b),
                 static_cast<const double*>(beta), static_cast<double*>(c),
                 rs_c, cs_c, aux);
}

struct blis_pack_ctx
{
    const gotobench::blis_kernels* ker = nullptr;
    bool is_a = true;
};

inline void blis_pack(void* ctx, dim_t dim, dim_t len, const void* kappa,
                      const void* src, inc_t inc, inc_t ld, void* dst, inc_t ldp)
{
    (void)kappa;   // the harness packs with kappa = 1
    (void)ldp;     // blis_kernels derives it from the kernel's panel width
    auto* p = static_cast<blis_pack_ctx*>(ctx);
    if (p->is_a)
        p->ker->pack_a(dim, len, static_cast<const double*>(src), inc, ld,
                       static_cast<double*>(dst));
    else
        p->ker->pack_b(dim, len, static_cast<const double*>(src), inc, ld,
                       static_cast<double*>(dst));
}

/// the BLIS micro-kernel as this nest's leaf
inline ukernel blis_kernel(const gotobench::blis_kernels& ker, blis_gemm_ctx& ctx)
{
    ukernel u;
    u.mr = ker.mr;
    u.nr = ker.nr;
    u.dt_a = u.dt_b = u.dt_c = ptype::f64;
    u.fn = &blis_gemm;
    u.ctx = &ctx;
    u.impl = ker.impl ? std::string(ker.impl) : std::string("BLIS");
    return u;
}

} // namespace momms
