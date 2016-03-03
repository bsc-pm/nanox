
#include "cachecommand.hpp"

#include "allocate.hpp"
#include "copydevtodev.hpp"
#include "copyin.hpp"
#include "copyout.hpp"
#include "free.hpp"
#include "realloc.hpp"

using namespace nanos::mpi::command;

MPI_Datatype CachePayload::_type = 0;

template <> 
BaseServant* BaseServant::createSpecific( CachePayload const& data )
{
	switch( data.getId() ) {
		case Allocate::id:
			return new Allocate::Servant( data );
		case CopyDeviceToDevice::id:
			return new CopyDeviceToDevice::Servant( data );
		case CopyIn::id:
			return new CopyIn::Servant( data );
		case CopyOut::id:
			return new CopyOut::Servant( data );
		case Free::id:
			return new Free::Servant( data );
		case Realloc::id:
			return new Realloc::Servant( data );
		default:
			fatal_error0( "Invalid nanos::mpi::CacheCommand id" );
	}
	return NULL;
}

