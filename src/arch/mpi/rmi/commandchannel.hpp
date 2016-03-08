
#ifndef COMMAND_SENDER_HPP
#define COMMAND_SENDER_HPP

#include <mpi.h>
#include "memoryaddress.hpp"

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
		}

		CommandChannel( int source, int destination, MPI_Comm communicator ) :
			_source( source ), _destination( destination ),
			_communicator( communicator )
		{
		}

		CommandChannel( MPIProcessor const& destination ) :
			_source( MPI_ANY_SOURCE ), _destination( destination.getRank() ),
			_communicator( destination.getCommunicator() )
		{
		}

		CommandChannel( MPIProcessor const& source, MPIProcessor const& destination ) :
			_source( source.getRank() ), _destination( destination.getRank() ),
			_communicator( source.getCommunicator() )
		{
			// TODO: ensure both source and destination communicators are the same
		}

		template < typename OldPayload, int other_tag >
		CommandChannel( CommandChannel<command_id,OldPayload,other_tag> const& other ) :
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

		int getTag() const
		{
			return channel_tag;
		}

		request isend( Payload const& data, size_t n = 1 );

		void receive( Payload &data, size_t n = 1 );

		void send( Payload const& data, size_t n = 1 );
};

template< int command_id, typename Payload, int tag >
void CommandChannel<command_id,Payload,tag>::receive( Payload &data, size_t n )
{
	MPIRemoteNode::nanosMPIRecv( &data, n, Payload::getDataType(),
	        getSource(), getTag(), getCommunicator(), MPI_STATUS_IGNORE );
}

template< int command_id, typename Payload, int tag >
void CommandChannel<command_id,Payload,tag>::send( Payload const& data, size_t n )
{
	MPIRemoteNode::nanosMPISend( &data, n, Payload::getDataType(),
	        getDestination(), getTag(), getCommunicator() );
}

template< int command_id, typename Payload, int tag >
request CommandChannel<command_id,Payload,tag>::isend( Payload const& data, size_t n )
{
	request result;
	MPIRemoteNode::nanosMPIIsend( &data, n, Payload::getDataType(),
	        getDestination(), getTag(), getCommunicator(), result );
	return result;
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COMMAND_SENDER_HPP
