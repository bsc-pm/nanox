
#include "command.hpp"

#include "createauxthread.hpp"
#include "finish.hpp"
#include "init.hpp"

#include "debug.hpp"

namespace nanos {
namespace mpi {
namespace command {

MPI_Datatype CommandPayload::_type = 0;

template <>
BaseServant* BaseServant::createSpecific( int source, int destination, MPI_Comm communicator, CommandPayload const& data )
{
	switch( data.getId() ) {
		case CreateAuxiliaryThread::id:
		{
			CreateAuxiliaryThread::main_channel_type channel( source, destination, communicator );
			return new CreateAuxiliaryThread::Servant( channel, data );
		}
		case Finish::id:
		{
			Finish::main_channel_type channel( source, destination, communicator );
			return new Finish::Servant( channel, data );
		}
		case Init::id:
		{
			Init::main_channel_type channel( source, destination, communicator );
			return new Init::Servant( channel, data );
		}
		default:
			fatal0( "Invalid nanos::mpi::Command id " << data.getId() );
	}
	// This point should never be reached
	return NULL;
}

} // namespace command
} // namespace mpi
} // namespace nanos

