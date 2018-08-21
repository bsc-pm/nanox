/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#ifndef PAGE_ALIGNED_ALLOCATOR_HPP
#define PAGE_ALIGNED_ALLOCATOR_HPP

#include "pagealignedallocator_decl.hpp"

#include <memory>
#include <new>

namespace nanos {
namespace mpi {

/**
 * Allocates memory with memory page alignment restriction
 * This may be useful for data transfers via DMA.
 *
 * By default, the threshold is a single cache line.
 * If no type is specified, byte sized allocation will be used.
 * This could produce performance degradation if the actual object
 * has a stricter alignment restriction, so it is not recommended
 * if the type is known.
 */
template <class T = char, size_t threshold = _NANOS_CACHE_LINESIZE >
struct PageAlignedAllocator {
	typedef T value_type;
	
	PageAlignedAllocator() throw() {}
	
	template <class U> PageAlignedAllocator (const PageAlignedAllocator<U>&) throw()
	{
	}
	
	T* allocate (std::size_t n)
	{
		T* result = NULL;
		size_t bytes = n*sizeof(T);
		if( bytes > threshold ) {
			posix_memalign( reinterpret_cast<void**>(&result), _NANOS_PAGESIZE, bytes );
		} else {
			//result = static_cast<T*>(::new(bytes));
			//posix_memalign( &result, alignof(T), bytes );
			result = static_cast<T*>(std::malloc(bytes));
		}

		if( result == NULL )
			throw std::bad_alloc();
		else
			return result;
	}
	
	void deallocate (T* p, std::size_t n = 0)
	{
		std::free( p );
	}
};

template <class T, size_t th1, class U, size_t th2>
bool operator== (const PageAlignedAllocator<T,th1>&, const PageAlignedAllocator<U,th2>&) throw()
{
	return true;
}

template <class T, size_t th1, class U, size_t th2>
bool operator!= (const PageAlignedAllocator<T,th1>&, const PageAlignedAllocator<U,th2>&) throw()
{
	return false;
}

} // namespace mpi
} // namespace nanos

#endif // PAGE_ALIGNED_ALLOCATOR_HPP

