
#include "cachecommand.hpp"

#include "allocate.hpp"
#include "copydevtodev.hpp"
#include "copyin.hpp"
#include "copyout.hpp"
#include "createauxthread.hpp"
#include "finish.hpp"
#include "free.hpp"
#include "init.hpp"
#include "realloc.hpp"

using namespace nanos::mpi::command;

MPI_Datatype GenericCacheCommand::_type = 0;

GenericCacheCommand* GenericCacheCommand::createSpecific( GenericCacheCommand &command )
{
	switch( command.getId() ) {
		case Allocate::id:
			return new Allocate::Servant( command );
		case CopyDeviceToDevice::id:
			return new CopyDeviceToDevice::Servant( command );
		case CopyIn::id:
			return new CopyIn::Servant( command );
		case CopyOut::id:
			return new CopyOut::Servant( command );
		case Free::id:
			return new Free::Servant( command );
		case Realloc::id:
			return new Realloc::Servant( command );
		case OPID_INVALID:
		default:
			fatal_error0( "Invalid nanos::mpi::CacheCommand id" );
	}
	return NULL;
}

