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

#ifndef FREE_HPP
#define FREE_HPP

#include "cachecommand.hpp"
#include "pagealignedallocator.hpp"

namespace nanos {
namespace mpi {
namespace command {

typedef CacheCommand<OPID_FREE> Free;

/**
 * No additional action required.
 */
template<>
inline void Free::Requestor::dispatch()
{
}

/**
 * Free specified memory 
 */
template<>
inline void Free::Servant::serve()
{
	NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_FREE_EVENT);

	PageAlignedAllocator<> allocator;
	allocator.deallocate( getData().getDeviceAddress(), getData().size() );

//	TODO: this might need an update to nanos cache v0.9
//
//	DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
//	if (ent != NULL)
//	{
//	    ent->setInvalidated(true);
//	}

	NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // FREE_HPP

