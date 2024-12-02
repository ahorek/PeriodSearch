#pragma once

#include <memory>
#include <cstdlib>
#include <stdexcept>
#include <new> // for placement new

// NOTE: To ensure alignment during dynamic allocation we will create a custom aligned allocator
template <typename T>
struct AlignedAllocator
{
    using value_type = T;
    std::size_t alignment;

    //AlignedAllocator() = default;
    explicit AlignedAllocator(std::size_t align = alignof(T)) noexcept : alignment(align) {}

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U>& other) noexcept : alignment(other.alignment) {}

    T* allocate(const std::size_t n)
    {
        void* ptr = _aligned_malloc(n * sizeof(T), alignment);
        if (!ptr)
        {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept
    {
        _aligned_free(p);
    }
};

template <typename T>
struct AlignedDeleter
{
    void operator()(T* ptr) const
    {
        ptr->~T();			// Explicitly call the destructor
        _aligned_free(ptr); // Free the aligned memory
    }
};

template <typename T, typename U>
bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return true; }

template <typename T, typename U>
bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return false; }

template <typename T>
std::shared_ptr<T> CreateAlignedShared(std::size_t alignment)
{
    // Allocate aligned memory and construct the object using placement new
    auto ptr = std::shared_ptr<T>(new (AlignedAllocator<T>(alignment).allocate(1)) T(), AlignedDeleter<T>());
    return ptr;
}

