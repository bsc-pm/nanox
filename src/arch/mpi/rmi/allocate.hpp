
#ifndef ALLOCATE_HPP
#define ALLOCATE_HPP

#include "cachecommand.hpp"
#include "commandchannel.hpp"

#include "mpidevice.hpp" // OPID definitions

#include "pagealignedallocator.hpp"

namespace nanos {
namespace mpi {
namespace command {

struct Allocate : public CacheCommand<OPID_ALLOCATE> {
	typedef CommandChannel<OPID_ALLOCATE, CachePayload, TAG_CACHE_ANSWER_ALLOC> ack_channel_type;
};

/**
 * Receives the updated information with final device address.
 */
template<>
inline void Allocate::Requestor::dispatch()
{
	// Destination and source are swapped
	Allocate::ack_channel_type ack( _channel.getDestination(), _channel.getSource(), _channel.getCommunicator() );
	ack.receive( getData() );
}

/**
 * Dynamically allocates memory and sends back the result.
 */
template<>
inline void Allocate::Servant::serve()
{
	NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_ALLOC_EVENT);

	PageAlignedAllocator<> allocator;
	getData().setDeviceAddress( allocator.allocate( getData().size() ) );

	Allocate::ack_channel_type ack( _channel.getDestination(), _channel.getSource(), _channel.getCommunicator() );
	ack.send( getData() );

	NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // ALLOCATE_HPP

