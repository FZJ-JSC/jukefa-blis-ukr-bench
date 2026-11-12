#pragma once
// A packer materialises one operand's blocks in the micro-panel layout the
// leaf kernel reads.  A block is `extent` elements along the operand's panel
// dimension (m for A, n for B) by `len` along k, laid out as ceil(extent/panel)
// micro-panels of panel x len, `ldp = panel*bcast` elements per k column.
//
// The kernel that does the moving is a plain function pointer with the shape
// every micro-panel packer has -- dimensions first, the datatype behind void*:
//
//     fn(ctx, dim, len, kappa, src, inc, ld, dst, ldp)
//
// BLIS's packm kernel and ukrgen's blisgen pack kernels both adapt to it, so a
// generated one drops in by address.  The packer needs the element size only
// to size its buffers and advance the source, never to do arithmetic.
//
// Nothing here includes a kernel family's headers: the function pointer, the
// element type and the layout below are mmgen's own.

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>

#include "allocator.hpp"
#include "types.hpp"

namespace mmgen {

// One micro-panel, as a pack kernel takes it.  `dim` valid entries along the
// panel dimension are written, padded to `panel`; `len` columns of the source
// are read with `inc` along the panel dimension and `ld` along k; the
// destination has `ldp` elements per column.  `kappa` scales if non-null.
using packm_fn = void (*)(void* ctx, index_t dim, index_t len, const void* kappa,
                          const void* src, stride_t inc, stride_t ld, void* dst,
                          stride_t ldp);

/// How one operand's blocks are laid out, as the provider's kernels expect:
/// the panel width, the broadcast factor, and the strides between k columns
/// and between panels.
struct packed_layout
{
    index_t panel = 0;        // micro-panel width: mr for A, nr for B
    index_t bcast = 1;        // broadcast factor of the packed layout
    index_t ldp = 0;          // elements per k column of one panel
    index_t panel_stride = 0; // elements from one micro-panel to the next
    ptype dt = ptype::f64;    // element type of this operand
};

class packer
{
public:
    struct shape
    {
        index_t panel = 0;    // micro-panel width: mr for A, nr for B
        index_t bcast = 1;    // broadcast factor of the packed layout
        index_t len = 0;      // k extent of one block, elements
        index_t extent = 0;   // panel-dimension extent of one block, elements
        ptype dt = ptype::f64;
        std::size_t slots = 1;
        std::size_t align = 64;   // the kernel's own alignment, rounded up by the allocator
    };

    packer(shape s, packm_fn fn, void* ctx, buffer_allocator& alloc)
        : s_(s), fn_(fn), ctx_(ctx),
          buf_(static_cast<char*>(alloc.allocate(block_bytes() * s_.slots, s.align)))
    {
        if (s_.panel <= 0 || s_.len <= 0 || s_.extent <= 0 || s_.slots == 0)
            throw std::invalid_argument("packer: empty block");
    }

    packer(const packer&) = delete;
    packer& operator=(const packer&) = delete;

    std::size_t slots() const noexcept { return s_.slots; }
    ptype dt() const noexcept { return s_.dt; }
    index_t extent() const noexcept { return s_.extent; }
    index_t len() const noexcept { return s_.len; }
    /// elements per k column of a panel
    index_t ldp() const noexcept { return s_.panel * s_.bcast; }
    /// elements from one micro-panel to the next
    index_t panel_stride() const noexcept { return ldp() * s_.len; }

    /// what the provider's kernel has to know about this buffer
    packed_layout layout() const noexcept
    {
        return packed_layout{s_.panel, s_.bcast, ldp(), panel_stride(), s_.dt};
    }

    /// one block: panels of panel x len, whether or not the block is full
    std::size_t block_elems() const noexcept
    {
        return static_cast<std::size_t>(panels()) * static_cast<std::size_t>(panel_stride());
    }
    std::size_t block_bytes() const noexcept { return block_elems() * elem_size(s_.dt); }

    /// Start of a block.  The register loops read panels from here, one
    /// panel_stride() apart.
    void* buffer(std::size_t slot) const
    {
        if (slot >= s_.slots) throw std::out_of_range("packer: slot");
        return buf_ + slot * block_bytes();
    }

    /// Pack one block from `src`; `extent` < shape.extent and `len` <
    /// shape.len are the partial last block of a dimension, and the layout is
    /// unchanged -- the padding is never read, because the kernel is called
    /// with the same partial extents.
    void pack(std::size_t slot, index_t extent, index_t len, const void* src,
              stride_t inc, stride_t ld) const
    {
        const std::size_t es = elem_size(s_.dt);
        char* dst = static_cast<char*>(buffer(slot));
        for (index_t p = 0, n = panels_of(extent); p < n; p++)
        {
            const index_t dim = std::min(s_.panel, extent - p * s_.panel);
            const char* from = static_cast<const char*>(src) + p * s_.panel * inc * es;
            char* to = dst + p * panel_stride() * es;
            if (fn_)
                fn_(ctx_, dim, len, nullptr, from, inc, ld, to, ldp());
            else
                copy_panel(dim, len, from, inc, ld, to, es);
        }
    }

private:
    index_t panels() const noexcept { return panels_of(s_.extent); }
    index_t panels_of(index_t extent) const noexcept
    {
        return extent / s_.panel + (extent % s_.panel != 0);
    }

    // No pack kernel for this type/layout: copy, broadcast on the way in,
    // zero-pad the panel.  Same result as the reference packm, slower.
    void copy_panel(index_t dim, index_t len, const char* src, stride_t inc, stride_t ld,
                    char* dst, std::size_t es) const
    {
        for (index_t l = 0; l < len; l++)
            for (index_t d = 0; d < s_.panel; d++)
                for (index_t b = 0; b < s_.bcast; b++)
                {
                    char* to = dst + ((l * ldp() + d * s_.bcast + b) * es);
                    if (d < dim)
                        std::memcpy(to, src + (d * inc + l * ld) * es, es);
                    else
                        std::memset(to, 0, es);
                }
    }

    shape s_;
    packm_fn fn_;
    void* ctx_;
    char* buf_;
};

} // namespace mmgen
