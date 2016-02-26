
#ifndef REALLOC_HPP
#define REALLOC_HPP

#include "cachecommand.hpp"

namespace nanos {
namespace mpi {
namespace command {

#if 0

class Realloc : public CacheCommand<OPID_REALLOC> {
	public:
		Realloc( MPIProcessor const& destination, Address deviceAddress, size_t size ) :
			CacheCommand( destination, Address::uninitialized(), deviceAddress, size )
		{
		}

		virtual ~Realloc()
		{
		}

		virtual void dispatch()
		{
    		MPIRemoteNode::nanosMPISend( static_cast<GenericCacheCommand*>(this), 1, GenericCacheCommand::getDataType(), getDestinationRank(), TAG_M2S_ORDER, getCommunicator() );
    		MPIRemoteNode::nanosMPIRecv( static_cast<GenericCacheCommand*>(this), 1, GenericCacheCommand::getDataType(), getDestinationRank(), TAG_CACHE_ANSWER_REALLOC, getCommunicator(), MPI_STATUS_IGNORE );
		}

		virtual void serve()
		{
			NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_REALLOC_EVENT);
			std::free((char *) order.devAddr);
			char* ptr;
			ptr = NULL;                        
			if (order.size<alignThreshold){
				ptr = (char*) malloc(order.size);
			} else {
				posix_memalign((void**)&ptr,alignment,order.size);
			}
//			DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) ptr );
//			if (ent != NULL) 
//			{ 
//				if (ent->getOwner() != NULL) 
//				{
//					ent->getOwner()->deleteEntry((uint64_t) ptr, order.size);
//				}
//			}
			setDeviceAddress( ptr );
			MPIRemoteNode::nanosMPISend( this, 1, MPIDevice::cacheStruct, getSourceRank(), TAG_CACHE_ANSWER_REALLOC, getCommunicator() );
			NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
		}
};

#endif

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // FREE_HPP

