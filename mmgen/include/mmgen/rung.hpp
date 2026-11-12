#pragma once
// One loop of a GEMM nest, the state a walk of the nest carries, and the value
// types everything else speaks: dimensions, operands, extents, strides.
//
// A nest is a list of loop rungs, outer to inner, then the two register loops
// around the micro-kernel.  A rung cuts one dimension -- m, n or k -- and may
// carry a packer for one operand, whose packed block the loops below it read.
//
// Nothing here knows about BLIS or about any particular kernel: a micro-kernel
// is reached through mmgen::ukr_provider, which also owns the packed layout's
// shaping (panel stride, schema, per-call hints).  A ukrgen kernel and a BLIS
// kernel are then two implementations of the same interface, and the model the
// nest walks does not change between them.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>

#include "types.hpp"

namespace mmgen {

enum class dimension : std::int64_t
{
    M = 0,
    N = 1,
    K = 2
};

inline constexpr std::array<dimension, 3> dimensions =
{
    dimension::M,
    dimension::N,
    dimension::K
};

enum class operand : std::int64_t
{
    A = 0,
    B = 1,
    C = 2
};

inline constexpr std::array<operand, 3> operands =
{
    operand::A,
    operand::B,
    operand::C
};

/// the operand a loop over a dimension leaves invariant: m keeps B, n keeps A,
/// k keeps C
inline constexpr std::array<operand, 3> dimension_invariants =
{{
    operand::B,
    operand::A,
    operand::C
}};

/// the two dimensions each operand spans
inline constexpr std::array<std::array<dimension, 2>, 3> operand_dimensions =
{{
    {dimension::M, dimension::K},
    {dimension::K, dimension::N},
    {dimension::M, dimension::N}
}};

inline constexpr std::array<char, 3> operand_names = {{'A', 'B', 'C'}};
inline constexpr std::array<char, 3> dimension_names = {{'M', 'N', 'K'}};
/// the spelling --nest takes: dimensions are written lower case there
inline constexpr std::array<char, 3> dimension_keys = {{'m', 'n', 'k'}};

inline constexpr std::size_t dim_index(dimension dim) noexcept
{
    return static_cast<std::size_t>(dim);
}

inline constexpr std::size_t op_index(operand op) noexcept
{
    return static_cast<std::size_t>(op);
}

/// the entry of a three-wide array belonging to `dim`; the const overload is
/// what a const query (m_exec() and friends) lands on
template<typename choice_type>
inline constexpr choice_type& index_by_dim(std::array<choice_type, 3>& view, dimension dim)
{
    return view[dim_index(dim)];
}

template<typename choice_type>
inline constexpr const choice_type& index_by_dim(const std::array<choice_type, 3>& view,
                                                 dimension dim)
{
    return view[dim_index(dim)];
}

/// one of three values, chosen by `dim`
template<typename choice_type>
inline constexpr choice_type& select_by_dim(choice_type& m_choice, choice_type& n_choice,
                                            choice_type& k_choice, dimension dim)
{
    if (dim == dimension::M) return m_choice;
    if (dim == dimension::N) return n_choice;
    return k_choice;
}

template<typename choice_type>
inline constexpr const choice_type& select_by_dim(const choice_type& m_choice,
                                                  const choice_type& n_choice,
                                                  const choice_type& k_choice, dimension dim)
{
    if (dim == dimension::M) return m_choice;
    if (dim == dimension::N) return n_choice;
    return k_choice;
}

/// the operand a loop over `dim` leaves invariant
inline constexpr operand invariant_of(dimension dim) noexcept
{
    return dimension_invariants[dim_index(dim)];
}

/// the two dimensions an operand spans
inline constexpr std::span<const dimension, 2> dims_of(operand op) noexcept
{
    return operand_dimensions[op_index(op)];
}

inline constexpr char op_name(operand op) noexcept { return operand_names[op_index(op)]; }
inline constexpr char dim_name(dimension dim) noexcept { return dimension_names[dim_index(dim)]; }
inline constexpr char dim_key(dimension dim) noexcept { return dimension_keys[dim_index(dim)]; }

inline constexpr index_t ndiv(index_t extent, index_t block) noexcept
{
    return extent / block + (extent % block != 0);
}

/// the size of iteration `i` of a loop cutting `extent` into `block`s
inline constexpr index_t segment(index_t extent, index_t block, index_t i) noexcept
{
    return std::min(block, extent - i * block);
}

/// the operand index in the packed-operand pair: A is 0, B is 1 (not C's 2)
inline constexpr std::size_t packed_index(operand op) noexcept { return op_index(op); }
inline constexpr bool is_packed(operand op) noexcept { return op != operand::C; }

// The most rungs a nest may have: the two register loops plus ten cache loops,
// more than any member of the MOMMS family needs (a dimension cut twice makes
// six).
constexpr int MAX_RUNGS = 12;

struct loop_rung
{
    dimension dim = dimension::M;   // the dimension it cuts
    index_t block = 0;              // its block size
    std::string label;              // 'jc', 'pc', 'ic', or n1, k1, m1, k2 ...
    int packs = 0;                  // bit 0: A is packed on entry, bit 1: B

    // One iteration moves C by this many elements: a cache loop that cuts m or
    // n walks C, and nothing else, since A and B are packed.
    stride_t delta_c = 0;

    // One iteration moves the source of A or B by this many elements.  Only a
    // deeper rung's packer uses these, to find the block it packs: the loops
    // below a packer read packed blocks instead.
    std::array<stride_t, 2> src_delta = {0, 0};

    bool is_register() const noexcept { return label == "ir" || label == "jr"; }
    bool packs_op(operand op) const noexcept
    {
        return is_packed(op) && (packs & (1 << packed_index(op)));
    }
};

struct state
{
    std::array<index_t, 3> ext = {0, 0, 0};   // the current segment's extent, per dimension
    std::array<std::size_t, MAX_RUNGS> idx = {};
    std::array<std::size_t, 2> slot = {0, 0}; // packed block in use, per operand
    stride_t off_c = 0;                       // elements into C
    bool first_k = true;                      // no k block walked yet: beta applies

    index_t extent(dimension dim) const noexcept { return index_by_dim<index_t>(ext, dim); }
    index_t& extent(dimension dim) noexcept { return index_by_dim<index_t>(ext, dim); }
    std::size_t packed_slot(operand op) const noexcept { return slot[packed_index(op)]; }
    std::size_t& packed_slot(operand op) noexcept { return slot[packed_index(op)]; }
};

/// One micro-kernel call: everything the leaf needs, and nothing else.
struct kernel_call
{
    index_t m = 0, n = 0, k = 0;
    const void* alpha = nullptr;
    const void* a = nullptr;        // the packed A panel to read
    const void* b = nullptr;        // the packed B panel to read
    const void* beta = nullptr;
    void* c = nullptr;
    stride_t rs_c = 0, cs_c = 0;
    const void* next_a = nullptr;   // the panel the next call reads, if known
    const void* next_b = nullptr;   // (a hint the provider may use or ignore)
};

} // namespace mmgen
