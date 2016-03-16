
#ifndef PAGEALIGNEDALLOCATOR_DECL
#define PAGEALIGNEDALLOCATOR_DECL

namespace nanos {
namespace mpi {

// FIXME: move this to autoconf check
// as this may not be the same on another
// platform
#define _NANOS_PAGESIZE 4096
#define _NANOS_CACHE_LINESIZE 128

template < class T, size_t threshold >
struct PageAlignedAllocator;

template <class T, size_t th1, class U, size_t th2>
bool operator== (const PageAlignedAllocator<T,th1>&, const PageAlignedAllocator<U,th2>&) throw();

template <class T, size_t th1, class U, size_t th2>
bool operator!= (const PageAlignedAllocator<T,th1>&, const PageAlignedAllocator<U,th2>&) throw();

} // namespace mpi
} // namespace nanos

#endif // PAGEALIGNEDALLOCATOR_DECL
