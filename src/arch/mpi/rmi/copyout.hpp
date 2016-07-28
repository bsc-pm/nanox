
#ifndef COPY_OUT_HPP
#define COPY_OUT_HPP

#include "cachecommand.hpp"

namespace nanos {
namespace mpi {
namespace command {

struct CopyOut : public CacheCommand<OPID_COPYOUT> {
	typedef CommandChannel<OPID_COPYOUT, RawPayload, TAG_CACHE_DATA_OUT> transfer_channel_type;
};

/**
 * Receive tasks's output data from the remote process
 */
template<>
inline void CopyOut::Requestor::dispatch()
{
	CopyOut::transfer_channel_type transfer_channel( _channel );
	transfer_channel.setSource( _channel.getDestination() );

	RawPayload hostData( _data.getHostAddress() );
	transfer_channel.receive( hostData, _data.size() );
}

/**
 * Send task's output data back to the master process
 */
template<>
inline void CopyOut::Servant::serve()
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

	CopyOut::transfer_channel_type transfer_channel( _channel );
	transfer_channel.setDestination( _channel.getSource() );

	RawPayload deviceData( _data.getDeviceAddress() );
	transfer_channel.send( deviceData, _data.size() );

	NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COPY_OUT_HPP

