
#ifndef COPY_DEV_TO_DEV_HPP
#define COPY_DEV_TO_DEV_HPP

#include "cachecommand.hpp"
#include "mpiremotenode.hpp"

#include <mpi.h>

namespace nanos {
namespace mpi {
namespace command {

typedef CacheCommand<OPID_DEVTODEV> CopyDeviceToDevice;

/**
 * sendCommand has to be redefined since we have to sent two messages
 * instead of one.
 * The messages are sent to both source and destination of the transfer
 */
template<>
void CopyDeviceToDevice::Requestor::sendCommand()
{
	MPI_Request req[2];
	MPIRemoteNode::nanosMPIIsend( static_cast<generic_type const*>(this),
	                  1, generic_type::getDataType(), getSourceRank(),
	                  TAG_M2S_ORDER, getCommunicator(), &req[0] );

	MPIRemoteNode::nanosMPIIsend( static_cast<generic_type const*>(this),
	                  1, generic_type::getDataType(), getDestinationRank(),
	                  TAG_M2S_ORDER, getCommunicator(), &req[1] );

	MPI_Waitall( 2, req, MPI_STATUSES_IGNORE );
}

/**
 * No additional actions required. Just send the message.
 */
template <>
void CopyDeviceToDevice::Requestor::dispatch()
{
}

/**
 * The process that matches with sourceRank has to send the data
 * to the one which matches with destinationRank
 */
template<>
void CopyDeviceToDevice::Servant::serve()
{
	int myRank;
	MPI_Comm_rank( getCommunicator(), &myRank );

	if ( myRank == getSourceRank() ) {
		// This process has to send the data
		NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_DEV2DEV_OUT_EVENT);

		MPIRemoteNode::nanosMPISend( getHostAddress(), size(), MPI_BYTE,
		                    getDestinationRank(), TAG_CACHE_DEV2DEV,
		                    getCommunicator() );

		NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;

	} else if ( myRank == getDestinationRank() ) {
		// This process has to receive the data
		NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_DEV2DEV_IN_EVENT);

//		TODO: this might need an update to nanos cache v0.9
//
//		DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
//		if (ent != NULL)
//		{
//		   if (ent->getOwner() != NULL) {
//		      ent->getOwner()->invalidate( *_masterDir, (uint64_t) order.devAddr, ent);
//		   } else {
//		      ent->increaseVersion();
//		   }
//		}

		MPIRemoteNode::nanosMPIRecv( getDeviceAddress(), size(), MPI_BYTE,
		                    getSourceRank(), TAG_CACHE_DEV2DEV,
		                    getCommunicator(), MPI_STATUS_IGNORE );

		NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
	}
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COPY_DEV_TO_DEV

