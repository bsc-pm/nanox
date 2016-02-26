
#ifndef COMMAND_REQUESTOR_HPP
#define COMMAND_REQUESTOR_HPP

#include "mpidevice_decl.hpp"
#include "commandchannel.hpp"

namespace nanos {
namespace mpi {
namespace command {

template< int command_id, typename Payload, typename Channel >
class CommandRequestor : public Channel {
	private:
		Payload _data;

	public:
		CommandRequestor() :
			Channel(),
			_data( command_id )
		{
			send( _data );
		}

		virtual ~CommandRequestor()
		{
		}

		Payload &getData()
		{
			return _data;
		}

		Payload const& getData() const
		{
			return _data;
		}

		/**
		 * Action to be executed after command is sent
		 * To be defined by each specific operation
		 */
		void dispatch();
};

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COMMAND_REQUESTOR_HPP

