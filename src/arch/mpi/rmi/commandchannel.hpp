
#ifndef COMMAND_SENDER_HPP
#define COMMAND_SENDER_HPP

#include <mpi.h>

namespace nanos {
namespace mpi {
namespace command {


// Inputchannel (only recv), Outputchannel (only send), InoutChannel (both)??

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

		int getSource() const
		{
			return _source;
		}

		int getDestination() const
		{
			return _destination;
		}

		MPI_Comm getCommunicator() const
		{
			return _communicator;
		}

		int getTag() const
		{
			return channel_tag;
		}

		void receive( Payload &data );

		void send( Payload const& data );
};

template< int command_id, typename Payload, int tag >
void CommandChannel<command_id,Payload,tag>::receive( Payload &data )
{
	MPIRemoteNode::nanosMPIRecv( &data, 1, Payload::getDataType(),
	        getSource(), getTag(), getCommunicator() );
}

template< int command_id, typename Payload, int tag >
void CommandChannel<command_id,Payload,tag>::send( Payload const& data )
{
	MPIRemoteNode::nanosMPISend( &data, 1, Payload::getDataType(),
	        getDestination(), getTag(), getCommunicator() );
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COMMAND_SENDER_HPP
