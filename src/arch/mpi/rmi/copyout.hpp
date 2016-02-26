
#ifndef COPY_OUT_HPP
#define COPY_OUT_HPP

#include "cachecommand.hpp"

namespace nanos {
namespace mpi {
namespace command {

typedef CacheCommand<OPID_COPYOUT> CopyOut;

/**
 * Receive tasks's output data from the remote process
 */
template<>
void CopyOut::dispatch()
{
	MPIRemoteNode::nanosMPIRecv( getHostAddress(), size(), MPI_BYTE,
	                    getDestinationRank(), TAG_CACHE_DATA_OUT,
	                    getCommunicator(), MPI_STATUS_IGNORE );
}

/**
 * Send task's output data back to the master process
 */
template<>
void CopyOut::serve()
{
	NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_COPYOUT_EVENT);

//	TODO: this might need an update to nanos cache v0.9
//
//	DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
//	if (ent != NULL)
//	{
//	   if (ent->getOwner() != NULL )
//	      if ( !ent->isInvalidated() )
//	      {
//	         std::list<uint64_t> tagsToInvalidate;
//	         tagsToInvalidate.push_back( ( uint64_t ) order.devAddr );
//	         _masterDir->synchronizeHost( tagsToInvalidate );
//	      }
//	}

	MPIRemoteNode::nanosMPISend( getDeviceAddress(), size(), MPI_BYTE,
	                    getSourceRank(), TAG_CACHE_DATA_OUT, getCommunicator() );

	NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COPY_OUT_HPP

