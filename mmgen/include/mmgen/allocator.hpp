#pragma once
// Where a plan gets its buffers: the packed blocks, and anything else it owns
// rather than the caller.  An interface, so a caller can hand in an arena, a
// pool, a device allocation, or count what is used.
//
// The default is the dynamic aligned allocator next door: page-aligned by
// default, optionally 2 MiB- and MADV_HUGEPAGE-backed, with the same optional
// offset that allocator supports for skewing cache sets.

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <new>
#include <string>
#include <utility>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "dynamic_aligned_allocator.hpp"

namespace mmgen {

/// Byte buffers a plan allocates for itself.
class buffer_allocator
{
public:
    virtual ~buffer_allocator() = default;

    /// `bytes` at `alignment` (a power of two); may return more alignment
    virtual void* allocate(std::size_t bytes, std::size_t alignment) = 0;
    virtual void release(void* p) noexcept = 0;
    virtual const char* name() const noexcept { return "custom"; }

    std::size_t blocks() const noexcept { return blocks_; }
    std::size_t bytes() const noexcept { return bytes_; }

protected:
    void account(std::size_t bytes) noexcept
    {
        ++blocks_;
        bytes_ += bytes;
    }

private:
    std::size_t blocks_ = 0, bytes_ = 0;
};

/// The library's own: dynamic_aligned_allocator<std::byte>, page-aligned,
/// zeroed (so a partial block's padding is never stale), huge pages on request.
class aligned_allocator : public buffer_allocator
{
public:
    explicit aligned_allocator(std::size_t alignment = 4096, std::size_t offset = 0,
                               bool hugepages = false)
        : alloc_(hugepages ? std::max<std::size_t>(alignment, 2u << 20) : alignment, offset)
    {
    }

    void* allocate(std::size_t bytes, std::size_t alignment) override
    {
        const std::size_t align = std::max<std::size_t>({alignment, alloc_.get_alignment(), sizeof(void*)});
        dynamic_aligned_allocator<std::byte> a(align, alloc_.get_offset());
        void* p = a.allocate(bytes);
        std::memset(p, 0, bytes);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        if (alloc_.get_alignment() >= (2u << 20))
        {
            const std::size_t page = 2u << 20;
            madvise(p, (bytes + page - 1) / page * page, MADV_HUGEPAGE);
        }
#endif
        account(bytes);
        return p;
    }

    void release(void* p) noexcept override
    {
        // deallocate only needs the offset to find the block it came from
        dynamic_aligned_allocator<std::byte> a(alloc_.get_alignment(), alloc_.get_offset());
        a.deallocate(static_cast<std::byte*>(p), 0);
    }

    const char* name() const noexcept override
    {
        return alloc_.get_alignment() >= (2u << 20) ? "dynamic-aligned+hugepages" : "dynamic-aligned";
    }

private:
    dynamic_aligned_allocator<std::byte> alloc_;
};

} // namespace mmgen
