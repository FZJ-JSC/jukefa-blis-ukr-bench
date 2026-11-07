#pragma once
// One loop of a GEMM nest, the state a walk of the nest carries, and the leaf
// the walk ends in.
//
// A nest is a list of loop rungs, outer to inner, then the two register loops
// around the micro-kernel.  A rung cuts one dimension -- m, n or k -- and may
// carry a packer for one operand, whose packed block the loops below it read.
//
// The state a walk carries is deliberately small and operand-shaped rather
// than generic: three extents, the C offset, the per-rung indices (which --top
// and the prefetcher need), the packed block each operand is reading, and
// whether this is the first k block of a tile (beta's business).  A and B are always packed -- the BLIS micro-kernel
// reads packed panels and nothing else -- so their addresses are (packer,
// slot, register-loop panel), never a running pointer.
//
// The leaf is a ukernel: a function pointer with BLIS's gemm micro-kernel
// signature, whose operands are void* and whose context is opaque.  A ukrgen
// kernel drops in here, including one for element types BLIS does not have --
// mr/nr say how the packers must lay the panels out, and the datatypes say
// how big the elements are; nothing else in the harness does arithmetic on
// the operands.

extern "C" {
#include <blis.h>
}

#include <algorithm>
#include <cstddef>
#include <string>

#include "packer.hpp"

namespace momms {

enum dims : int { DIM_M = 0, DIM_N = 1, DIM_K = 2 };
enum ops : int { OP_A = 0, OP_B = 1, OP_C = 2 };

/// the operand a loop over `dim` leaves invariant
inline int invariant_of(int dim) { return dim == DIM_M ? OP_B : dim == DIM_N ? OP_A : OP_C; }

/// the two dimensions an operand spans
inline const int* dims_of(int op)
{
    static const int d[3][2] = {{DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_M, DIM_N}};
    return d[op];
}

inline const char* op_name(int op) { return op == OP_A ? "A" : op == OP_B ? "B" : "C"; }
inline const char* dim_name(int dim) { return dim == DIM_M ? "m" : dim == DIM_N ? "n" : "k"; }

inline dim_t ndiv(dim_t extent, dim_t block) { return extent / block + (extent % block != 0); }

/// the size of iteration `i` of a loop cutting `extent` into `block`s
inline dim_t segment(dim_t extent, dim_t block, dim_t i)
{
    return std::min(block, extent - i * block);
}

// The most rungs a nest may have: the two register loops plus ten cache loops,
// more than any member of the MOMMS family needs (a dimension cut twice makes
// six).
constexpr int kMaxRungs = 12;

struct loop_rung
{
    int dim = DIM_M;              // the dimension it cuts
    dim_t block = 0;              // its block size
    std::string label;            // 'jc', 'pc', 'ic', or n1, k1, m1, k2 ...
    int packs = 0;                // bit 0: A is packed on entry, bit 1: B

    // One iteration moves C by this many elements: a cache loop that cuts m or
    // n walks C, and nothing else, since A and B are packed.
    std::ptrdiff_t delta_c = 0;

    // One iteration moves the source of A or B by this many elements.  Only a
    // deeper rung's packer uses these, to find the block it packs: the loops
    // below a packer read packed blocks instead.
    std::ptrdiff_t src_delta[2] = {0, 0};

    bool is_register() const { return label == "ir" || label == "jr"; }
};

struct state
{
    dim_t ext[3] = {0, 0, 0};     // extent of the current segment: m, n, k
    std::size_t idx[kMaxRungs] = {};
    std::size_t slot[2] = {0, 0}; // packed block in use, kept by the walk
    std::ptrdiff_t off_c = 0;     // elements into C
    bool first_k = true;          // no k block has been walked yet: beta applies
};

// The micro-kernel the nest ends in.  The signature is BLIS's (>= 1.0), with
// the context opaque: an adapter for a ukrgen kernel is a short function that
// knows its own argument shapes and calls it.
using gemm_fn = void (*)(void* ctx, dim_t m, dim_t n, dim_t k, const void* alpha,
                         const void* a, const void* b, const void* beta, void* c,
                         inc_t rs_c, inc_t cs_c, auxinfo_t* aux);

struct ukernel
{
    dim_t mr = 0, nr = 0;
    ptype dt_a = ptype::f64, dt_b = ptype::f64, dt_c = ptype::f64;
    gemm_fn fn = nullptr;
    void* ctx = nullptr;
    std::string impl;             // for the report: BLIS's, or which ukrgen kernel
};

} // namespace momms
