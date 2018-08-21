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

#ifndef REALLOC_HPP
#define REALLOC_HPP

#include "cachecommand.hpp"
#include "memoryaddress.hpp"

namespace nanos {
namespace mpi {
namespace command {

struct Realloc : public CacheCommand<OPID_REALLOC> {
	typedef CommandChannel<OPID_REALLOC, CachePayload, TAG_CACHE_ANSWER_REALLOC> ack_channel_type;
};


template<>
inline void Realloc::Requestor::dispatch()
{
	// Destination and source are swapped
	Allocate::ack_channel_type ack( _channel.getDestination(), _channel.getSource(), _channel.getCommunicator() );
	ack.receive( getData() );
}

/*
 * Note: Nanos cache does not support realloc currently.
 * However, an example implementation is provided for this device.
 * It assumes memory region is contiguous.
 * It cannot use realloc because the new region alignment would not
 * be warrantee to be the same (in case memory block can not be expanded).
 */
template<>
inline void Realloc::Servant::serve()
{
	NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_REALLOC_EVENT);

	PageAlignedAllocator<> allocator;
	utils::Address newAddress( allocator.allocate( getData().size() ) );

	// TODO: Copy the old data in the new region	(we would need to query
	// the region dictionary or the device cache in some way ).

	allocator.deallocate( getData().getDeviceAddress() );

//	DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) ptr );
//	if (ent != NULL) 
//	{ 
//		if (ent->getOwner() != NULL) 
//		{
//			ent->getOwner()->deleteEntry((uint64_t) ptr, order.size);
//		}
//	}

	getData().setDeviceAddress( newAddress );
	Realloc::ack_channel_type ack( _channel.getDestination(), _channel.getSource(), _channel.getCommunicator() );
	ack.send( getData() );

	NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // FREE_HPP

