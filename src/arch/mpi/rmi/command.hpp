
#ifndef COMMAND_HPP
#define COMMAND_HPP

#include "commandid.hpp"
#include "commandpayload.hpp"

#include "memoryaddress.hpp"
#include "mpidevice_decl.hpp"
#include "mpiprocessor_decl.hpp"

namespace nanos {
namespace mpi {
namespace command {

using namespace ext;

template < int id, typename Channel >
class CommandRequestor< id, CommandPayload, Channel > {
	private:
		CommandPayload _data;
		Channel        _channel;

	public:
		CommandRequestor( MPIProcessor const& destination ) :
			_data(id), _channel( destination )
		{
			_channel.send( _data );
		}

		CommandRequestor( MPIProcessor const& destination, int code ) :
			_data(id, code), _channel( destination )
		{
			_channel.send( _data );
		}

		CommandRequestor( int destination, MPI_Comm communicator, int code ) :
			_data(id, code), _channel( destination, communicator )
		{
			_channel.send( _data );
		}

		virtual ~CommandRequestor()
		{
		}

		CommandPayload &getData()
		{
			return _data;
		}

		CommandPayload const& getData() const
		{
			return _data;
		}

		// To be defined by each operation
		void dispatch();
};

/**
 * Pairs Requestor and Servant types for each operation id
 */
template < int op_id, int tag = TAG_M2S_COMMAND >
struct Command {
	static const int id;

	typedef CommandPayload	                           payload_type;
	typedef CommandChannel<op_id, CommandPayload, tag> main_channel_type;

	typedef CommandRequestor< op_id, payload_type, main_channel_type > Requestor;
	typedef CommandServant  < op_id, payload_type, main_channel_type > Servant;
};

template< int op_id, int tag >
const int Command<op_id,tag>::id = op_id;

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COMMAND_HPP

