
#ifndef ALLOCATE_HPP
#define ALLOCATE_HPP

#include "cachecommand.hpp"
#include "commandchannel.hpp"
#include "mpidevice_decl.hpp"

namespace nanos {
namespace mpi {
namespace command {

struct Allocate {
	typedef CachePayload payload_type;
	typedef CacheCommand<OPID_ALLOCATE>::main_channel main_channel;
	typedef CommandChannel<OPID_ALLOCATE, CachePayload, TAG_CACHE_ANSWER_ALLOC> ack_channel;

	typedef CacheCommand<OPID_ALLOCATE>::Requestor Requestor;
	typedef CacheCommand<OPID_ALLOCATE>::Servant Servant;
};

/**
 * Helper function that just allocates memory.
 * The allocation is aligned whenever its size is greater
 * or equal to a given threshold.
 * Threshold can be set using NX_OFFL_ALIGNTHRESHOLD
 */
inline Address _allocate( size_t size )
{
	Address addr( Address::uninitialized() );
	if ( size >= MPIProcessor::getAlignThreshold() ) {
	   posix_memalign( addr, MPIProcessor::getAlignment(), size );
	} else {
	   addr = std::malloc( size );
	}
	return addr;
}

/**
 * Receives the updated information with final device address.
 */
template<>
void Allocate::Requestor::dispatch()
{
	// Destination and source are swapped
	Allocate::ack_channel ack( getDestination(), getSource(), getCommunicator() );
	ack.receive( getData() );
}

/**
 * Dynamically allocates memory and sends back the result
 */
template<>
void Allocate::Servant::serve()
{
	NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_ALLOC_EVENT);

	getData().setDeviceAddress( _allocate( getData().size() ) );

	Allocate::ack_channel ack( getDestination(), getSource(), getCommunicator() );
	ack.send( getData() );

	NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // ALLOCATE_HPP

