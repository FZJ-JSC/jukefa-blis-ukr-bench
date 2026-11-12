#pragma once
// BLIS as an mmgen provider: its micro-kernel (BLIS >= 1.0 or AOCL's 0.x
// interface) and its packm kernel, with everything BLIS-specific kept here --
// auxinfo, pack schemas, panel strides, next-panel hints -- so the nest never
// sees any of it.
//
// This is one ukr_provider.  A ukrgen adapter is another, and the core of the
// library compiles without BLIS's headers either way.

#include <string>

#include "blis_kernels.hpp"

#include "mmgen/gemm.hpp"
#include "mmgen/ukr_provider.hpp"

namespace mmgen {

class blis_provider final : public ukr_provider
{
public:
    explicit blis_provider(const gotobench::blis_kernels& ker)
        : ker_(ker), pack_a_{&ker, true}, pack_b_{&ker, false}
    {
        info_.mr = ker.mr;
        info_.nr = ker.nr;
        info_.dt_a = info_.dt_b = info_.dt_c = ptype::f64;
        info_.impl = ker.impl ? ker.impl : "BLIS";

        // constant for the plan's life; begin_plan adds the strides and schema
        bli_auxinfo_set_is_a(1, &aux_);
        bli_auxinfo_set_is_b(1, &aux_);
    }

    const kernel_info& info() const override { return info_; }

    /// drop the packm kernels: the plan then falls back to its own copy loop,
    /// which is what `--pack-kernel loop` measures
    void disable_packers() noexcept { no_pack_ = true; }

    pack_kernel packer(operand op) const override
    {
        if (no_pack_) return {};
        // BLIS's packm kernel knows its panel width from the context, so the
        // layout's ldp is implicit on that side
        pack_ctx* ctx = op == operand::A ? const_cast<pack_ctx*>(&pack_a_)
                                         : const_cast<pack_ctx*>(&pack_b_);
        return pack_kernel{&pack_panel, ctx};
    }

    void begin_plan(const packed_layout& a, const packed_layout& b) override
    {
        ps_a_ = a.panel_stride;
        ps_b_ = b.panel_stride;
#if defined(AOCL_BLIS)
        bli_auxinfo_set_schema_a(BLIS_PACKED_ROW_PANELS, &aux_);
        bli_auxinfo_set_schema_b(BLIS_PACKED_COL_PANELS, &aux_);
#else
        bli_auxinfo_set_schema_a(BLIS_PACKED_PANELS, &aux_);
        bli_auxinfo_set_schema_b(BLIS_PACKED_PANELS, &aux_);
#endif
        bli_auxinfo_set_ps_a(ps_a_, &aux_);
        bli_auxinfo_set_ps_b(ps_b_, &aux_);
    }

    void gemm(const kernel_call& c) override
    {
        // the panel hints a kernel may prefetch with; one that ignores them is
        // equally welcome
        bli_auxinfo_set_next_a(const_cast<char*>(static_cast<const char*>(c.next_a)), &aux_);
        bli_auxinfo_set_next_b(const_cast<char*>(static_cast<const char*>(c.next_b)), &aux_);
        ker_.gemm(static_cast<dim_t>(c.m), static_cast<dim_t>(c.n), static_cast<dim_t>(c.k),
                  static_cast<const double*>(c.alpha), static_cast<const double*>(c.a),
                  static_cast<const double*>(c.b), static_cast<const double*>(c.beta),
                  static_cast<double*>(c.c), c.rs_c, c.cs_c, &aux_);
    }

private:
    struct pack_ctx
    {
        const gotobench::blis_kernels* ker = nullptr;
        bool is_a = true;
    };

    static void pack_panel(void* ctx, index_t dim, index_t len, const void* kappa,
                           const void* src, stride_t inc, stride_t ld, void* dst,
                           stride_t ldp)
    {
        (void)kappa;   // the harness packs with kappa = 1
        (void)ldp;     // blis_kernels derives it from the kernel's panel width
        auto* p = static_cast<pack_ctx*>(ctx);
        if (p->is_a)
            p->ker->pack_a(static_cast<dim_t>(dim), static_cast<dim_t>(len),
                           static_cast<const double*>(src), inc, ld,
                           static_cast<double*>(dst));
        else
            p->ker->pack_b(static_cast<dim_t>(dim), static_cast<dim_t>(len),
                           static_cast<const double*>(src), inc, ld,
                           static_cast<double*>(dst));
    }

    const gotobench::blis_kernels& ker_;
    kernel_info info_;
    auxinfo_t aux_{};
    pack_ctx pack_a_, pack_b_;
    stride_t ps_a_ = 0, ps_b_ = 0;
    bool no_pack_ = false;
};

} // namespace mmgen
