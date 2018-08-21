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

#ifndef COMMAND_SENDER_HPP
#define COMMAND_SENDER_HPP

#include "memoryaddress.hpp"
#include "mpiprocessor_decl.hpp"

#include <mpi.h>

namespace nanos {
namespace mpi {
namespace command {

/**
 * This struct represents a payload of contiguous
 * memory. It is analogous to MPI_BYTE.
 */
class RawPayload {
	private:
		utils::Address _address;
	public:
		RawPayload( utils::Address const& address ) :
			_address( address )
		{
		}

		void* operator&() { return static_cast<void*>(_address); }

		const void* operator&() const { return static_cast<const void*>(_address); }

		static MPI_Datatype getDataType() { return MPI_BYTE; }
};

template< int command_id, typename Payload, int channel_tag >
class CommandChannel {
	private:
		int _source;
		int _destination;
		MPI_Comm _communicator;

	public:
		CommandChannel() :
			_source( MPI_ANY_SOURCE ), _destination( MPI_PROC_NULL ),
			_communicator( MPI_COMM_NULL )
		{
		}

		CommandChannel( int destination, MPI_Comm communicator ) :
			_source( MPI_ANY_SOURCE ), _destination( destination ),
			_communicator( communicator )
		{
			checkDestinationRank();
		}

		CommandChannel( int source, int destination, MPI_Comm communicator ) :
			_source( source ), _destination( destination ),
			_communicator( communicator )
		{
			checkDestinationRank();
		}

		CommandChannel( const ext::MPIProcessor& destination ) :
			_source( MPI_ANY_SOURCE ), _destination( destination.getRank() ),
			_communicator( destination.getCommunicator() )
		{
		}

		CommandChannel( const ext::MPIProcessor& source, const ext::MPIProcessor& destination ) :
			_source( source.getRank() ), _destination( destination.getRank() ),
			_communicator( source.getCommunicator() )
		{
			// TODO: ensure both source and destination communicators are the same
		}

		template < typename OldPayload, int other_tag >
		CommandChannel( const CommandChannel<command_id,OldPayload,other_tag>& other ) :
			_source( other.getSource() ), _destination( other.getDestination() ),
			_communicator( other.getCommunicator() )
		{
		}

		int getSource() const
		{
			return _source;
		}

		void setSource( int source )
		{
			_source = source;
		}

		int getDestination() const
		{
			return _destination;
		}

		void setDestination( int destination )
		{
			_destination = destination;
		}

		MPI_Comm getCommunicator() const
		{
			return _communicator;
		}

		int getId() const
		{
			return command_id;
		}

		int getTag() const
		{
			return channel_tag;
		}

		request isend( Payload const& data, size_t n = 1 );

		void receive( Payload &data, size_t n = 1 );

		void send( Payload const& data, size_t n = 1 );

		void checkDestinationRank()
		{
			using namespace nanos::ext;
			if( _destination == UNKNOWN_RANK ) {
				MPIProcessor& remote = *static_cast<MPIProcessor*>( myThread->runningOn() );
				_destination = remote.getRank();
				_communicator = remote.getCommunicator();
			}
		}
};

template< int command_id, typename Payload, int tag >
inline void CommandChannel<command_id,Payload,tag>::receive( Payload &data, size_t n )
{
	int err = MPI_Recv( &data, n, Payload::getDataType(),
	        getSource(), getTag(), getCommunicator(), MPI_STATUS_IGNORE );
	fatal_cond0( err != MPI_SUCCESS, "MPI_Recv finished with errors" );
}

template< int command_id, typename Payload, int tag >
inline void CommandChannel<command_id,Payload,tag>::send( Payload const& data, size_t n )
{
	int err = MPI_Send( &data, n, Payload::getDataType(),
	        getDestination(), getTag(), getCommunicator() );
	fatal_cond0( err != MPI_SUCCESS, "MPI_Send finished with errors" );
}

template< int command_id, typename Payload, int tag >
inline request CommandChannel<command_id,Payload,tag>::isend( Payload const& data, size_t n )
{
	request result;
	int err = MPI_Isend( &data, n, Payload::getDataType(),
	        getDestination(), getTag(), getCommunicator(), result );
	fatal_cond0( err != MPI_SUCCESS, "MPI_ISend finished with errors" );
	return result;
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COMMAND_SENDER_HPP
