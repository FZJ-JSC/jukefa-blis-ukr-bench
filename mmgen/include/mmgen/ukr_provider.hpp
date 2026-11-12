#pragma once
// The leaf of a plan: one micro-kernel, behind an interface.
//
// A plan is generated for one provider, and the provider owns everything that
// would otherwise leak a kernel family into the nest: the register blocking,
// the element types, the pack kernels, and the whole packed layout's shaping
// (panel strides, schemas, per-call hints).  The nest only ever says "call
// your kernel on these two panels and this tile".
//
// Implementations live wherever their kernels live -- BLIS's in harness/,
// ukrgen's next to ukrgen -- and the core of mmgen never includes their
// headers.

#include <string>

#include "packer.hpp"
#include "rung.hpp"
#include "types.hpp"

namespace mmgen {

/// what a plan needs to know about its leaf before it can build anything
struct kernel_info
{
    index_t mr = 0, nr = 0;                                // register blocking
    ptype dt_a = ptype::f64, dt_b = ptype::f64, dt_c = ptype::f64;
    std::string impl;                                      // for the report
};

/// a pack kernel for one operand, with the context it needs; `fn == nullptr`
/// means the packer's own copy loop
struct pack_kernel
{
    packm_fn fn = nullptr;
    void* ctx = nullptr;
};

class ukr_provider
{
public:
    virtual ~ukr_provider() = default;

    /// register blocking, element types, and a name for the report
    virtual const kernel_info& info() const = 0;

    /// the pack kernel for one operand, or {nullptr, nullptr} for the copy loop
    virtual pack_kernel packer(operand op) const = 0;

    /// The plan is built: these are the layouts every call will read.  The
    /// provider keeps whatever its kernel needs from them -- panel strides,
    /// schemas, register state -- and the nest never sees it again.
    virtual void begin_plan(const packed_layout& a, const packed_layout& b) = 0;

    /// one micro-kernel call: C(m x n) = alpha*A*B + beta*C over k
    virtual void gemm(const kernel_call& c) = 0;
};

} // namespace mmgen
