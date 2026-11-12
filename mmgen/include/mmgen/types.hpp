#pragma once
// The library's scalar vocabulary, independent of any kernel family: an index
// along a dimension, a distance between elements, and the element types a
// packed buffer can hold.  ukr_provider translates between these and whatever
// its kernels want (BLIS's dim_t/inc_t, a ukrgen kernel's own shapes).

#include <cstddef>
#include <cstdint>

namespace mmgen {

/// a matrix dimension's extent, or how far along it a loop has walked
using index_t = std::int64_t;

/// a distance between elements of one matrix, in elements
using stride_t = std::int64_t;

// The element types the harness can lay out.  A ukrgen kernel may bring its
// own (int8, bf16, ...): add it here with its size and the packing path needs
// no other change.
enum class ptype : int { f32 = 4, f64 = 8 };

inline constexpr std::size_t elem_size(ptype t) { return static_cast<std::size_t>(t); }

} // namespace mmgen
