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

#include "cachecommand.hpp"

#include "allocate.hpp"
#include "copydevtodev.hpp"
#include "copyin.hpp"
#include "copyout.hpp"
#include "free.hpp"
#include "realloc.hpp"

#include "debug.hpp"

#include <mpi.h>

namespace nanos {
namespace mpi {
namespace command {

MPI_Datatype CachePayload::_type = 0;

template <> 
BaseServant* BaseServant::createSpecific( int source, int destination, MPI_Comm communicator, CachePayload const& data )
{
	switch( data.getId() ) {
		case Allocate::id:
		{
			Allocate::main_channel_type channel( source, destination, communicator );
			return new Allocate::Servant( channel, data );
		}
		case CopyDeviceToDevice::id:
		{
			CopyDeviceToDevice::main_channel_type channel( source, destination, communicator );
			return new CopyDeviceToDevice::Servant( channel, data );
		}
		case CopyIn::id:
		{
			CopyIn::main_channel_type channel( source, destination, communicator );
			return new CopyIn::Servant( channel, data );
		}
		case CopyOut::id:
		{
			CopyOut::main_channel_type channel( source, destination, communicator );
			return new CopyOut::Servant( channel, data );
		}
		case Free::id:
		{
			Free::main_channel_type channel( source, destination, communicator );
			return new Free::Servant( channel, data );
		}
		case Realloc::id:
		{
			Realloc::main_channel_type channel( source, destination, communicator );
			return new Realloc::Servant( channel, data );
		}
		default:
			fatal0( "Invalid nanos::mpi::CacheCommand id: " << data.getId() );
	}
	// This point should never be reached
	return NULL;
}

} // namespace command
} // namespace mpi
} // namespace nanos

