
#ifndef COPYIN_HPP
#define COPYIN_HPP

#include "cachecommand.hpp"
#include "mpiremotenode.hpp"

#include <mpi.h>

namespace nanos {
namespace mpi {
namespace command {

typedef CacheCommand<OPID_COPYIN> CopyIn;

template <>
class CopyIn::Requestor : public CopyIn::RequestorBase {
	private:
		MPIProcessor &_remoteProcess;

	public:
		Requestor( MPIProcessor &destination, Address hostAddress, Address deviceAddress, size_t size ) :
			generic_type( destination, hostAddress, deviceAddress, size ),
			_remoteProcess( destination )
		{
			sendCommand();
		}

		virtual ~Requestor()
		{
		}

		void dispatch();
};

/**
 * Send the data to the remote process
 *
 * Appends the pending mpi::request to MPIProcessor request 
 * queue since we don't really need to use the send buffer inmediately.
 */
template<>
void CopyIn::Requestor::dispatch()
{
	MPI_Request req;
	MPIRemoteNode::nanosMPIIsend( getHostAddress(), size(),
	                  MPI_BYTE, getDestinationRank(),
	                  TAG_CACHE_DATA_IN, getCommunicator(), &req );

	_remoteProcess.appendToPendingRequests(req);
}

/**
 * Receive the data from the master process
 */
template<>
void CopyIn::Servant::serve()
{
	NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_COPYIN_EVENT);

	MPIRemoteNode::nanosMPIRecv( getDeviceAddress(), size(),
	                   MPI_BYTE, getSourceRank(),
	                   TAG_CACHE_DATA_IN, getCommunicator(), MPI_STATUS_IGNORE );

//	TODO: this might need an update to nanos cache v0.9
//	DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
//	if (ent != NULL)
//	{
//	   if (ent->getOwner() != NULL) {
//	      ent->getOwner()->invalidate( *_masterDir, (uint64_t) order.devAddr, ent);
//	   } else {
//	      ent->increaseVersion();
//	   }
//	}

	NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COPY_IN_HPP

