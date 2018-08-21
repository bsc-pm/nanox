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
