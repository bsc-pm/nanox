
#include "command.hpp"

#include "createauxthread.hpp"
#include "finish.hpp"
#include "init.hpp"

using namespace nanos::mpi::command;

GenericCommand::_type = 0;

GenericCommand* GenericCommand::createSpecific( GenericCommand &command )
{
	switch( command.getId() ) {
		case CreateAuxiliaryThread::id:
			return new CreateAuxiliaryThread::Servant( command );
		case Finish::id:
			return new Finish::Servant( command );
		case Init::id:
			return new Init::Servant( command );
		case OPID_INVALID:
		default:
			fatal_error0( "Invalid nanos::mpi::Command id" );
	}
}
