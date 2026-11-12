#ifndef MMGEN_DYNAMIC_ALIGNED_ALLOCATOR_HPP
#define MMGEN_DYNAMIC_ALIGNED_ALLOCATOR_HPP

// A std-style allocator with runtime alignment: for a std container, or for a
// byte buffer whose alignment the caller only learns at run time (a
// micro-kernel's packed panels, say).  An optional offset deliberately moves
// the returned pointer off the alignment boundary, to skew the cache sets of
// one buffer against another.
//
// Originally from an older project (namespace archcomp); now mmgen's.

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>

namespace mmgen
{

template<class T>
struct dynamic_aligned_allocator
{
    typedef T value_type;

    dynamic_aligned_allocator(std::size_t alignment, std::size_t offset = 0UL)
        : alignment(alignment), offset(offset)
    {
        if ((0 == alignment) || !((alignment & (alignment - 1)) == 0))
            throw std::invalid_argument("Alignment not a power of 2");
        if (offset >= alignment)
            throw std::invalid_argument("Offset must be smaller than the alignment");
    }

    template<class U>
    constexpr dynamic_aligned_allocator(const dynamic_aligned_allocator<U>& other) noexcept
        : alignment(other.alignment), offset(other.offset)
    {
    }

    [[nodiscard]] T* allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();

        // the offset is carved out of the front of an aligned block, so the
        // returned pointer sits `offset` bytes past an alignment boundary;
        // aligned_alloc wants a size that is a multiple of the alignment
        const std::size_t adjust = offset ? alignment : 0;
        const std::size_t bytes = adjust + n * sizeof(T);
        const std::size_t rounded = std::max(alignment, (bytes + alignment - 1) / alignment * alignment);

        if (auto p = static_cast<void*>(std::aligned_alloc(alignment, rounded)))
            return reinterpret_cast<T*>(static_cast<std::byte*>(p) + offset);

        throw std::bad_alloc();
    }

    void deallocate(T* p, std::size_t /*n*/) noexcept
    {
        std::free(reinterpret_cast<std::byte*>(p) - offset);
    }

    std::size_t get_alignment() const noexcept { return alignment; }
    std::size_t get_offset() const noexcept { return offset; }

private:
    std::size_t alignment;
    std::size_t offset;

    template<class U>
    friend bool operator==(const dynamic_aligned_allocator<T>& a1,
                           const dynamic_aligned_allocator<U>& a2);

    template<class U>
    friend bool operator!=(const dynamic_aligned_allocator<T>& a1,
                           const dynamic_aligned_allocator<U>& a2);
};

template<class T, class U>
bool operator==(const dynamic_aligned_allocator<T>& a1, const dynamic_aligned_allocator<U>& a2)
{
    return a1.get_alignment() == a2.get_alignment() && a1.get_offset() == a2.get_offset();
}

template<class T, class U>
bool operator!=(const dynamic_aligned_allocator<T>& a1, const dynamic_aligned_allocator<U>& a2)
{
    return !(a1 == a2);
}

} // namespace mmgen

#endif // MMGEN_DYNAMIC_ALIGNED_ALLOCATOR_HPP
