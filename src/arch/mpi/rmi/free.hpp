
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

