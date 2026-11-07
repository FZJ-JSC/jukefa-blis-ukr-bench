#pragma once
// Where the nest's buffers come from: the matrices, and every packer's packed
// blocks.  An interface, so a caller can put them on huge pages, in an arena,
// on a device, or count them -- the nest itself never calls malloc.
//
// The default is what the GOTO harness used: page-aligned (2 MiB-aligned and
// MADV_HUGEPAGE when asked), zeroed.

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace momms {

class allocator
{
public:
    virtual ~allocator() = default;

    /// `bytes` zeroed, aligned to `align` (at least a page); throws bad_alloc
    virtual void* allocate(std::size_t bytes, std::size_t align = 4096) = 0;
    virtual void release(void* p) noexcept = 0;
    virtual const char* name() const noexcept { return "aligned"; }
};

class aligned_allocator : public allocator
{
public:
    explicit aligned_allocator(bool hugepages = false) : hugepages_(hugepages) {}

    void* allocate(std::size_t bytes, std::size_t align = 4096) override
    {
        if (hugepages_) align = std::max<std::size_t>(align, 2u << 20);
        const std::size_t n = std::max<std::size_t>(bytes, align);
        void* p = nullptr;
        if (posix_memalign(&p, align, n) != 0) throw std::bad_alloc();
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        if (hugepages_) madvise(p, n, MADV_HUGEPAGE);
#endif
        std::memset(p, 0, n);
        blocks_ += 1;
        bytes_ += n;
        return p;
    }

    void release(void* p) noexcept override { std::free(p); }
    const char* name() const noexcept override { return hugepages_ ? "aligned+hugepages" : "aligned"; }

    /// what has been handed out, for a --alloc-stats line
    std::size_t blocks() const noexcept { return blocks_; }
    std::size_t bytes() const noexcept { return bytes_; }

private:
    bool hugepages_;
    std::size_t blocks_ = 0, bytes_ = 0;
};

} // namespace momms
