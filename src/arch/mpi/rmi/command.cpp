/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

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

