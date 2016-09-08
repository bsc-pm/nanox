
#ifndef COMMAND_REQUESTOR_HPP
#define COMMAND_REQUESTOR_HPP

#include "mpidevice.hpp"
#include "commandchannel.hpp"

namespace nanos {
namespace mpi {
namespace command {

template< int command_id, typename Payload, typename Channel >
class CommandRequestor {
	private:
		Payload _data;
		Channel _channel;

	public:
		CommandRequestor( Channel const& channel ) :
			_data(),
			_channel( channel )
		{
			_data.initialize(command_id);
			_channel.send( _data );
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

		Channel const& getChannel() const
		{
			return _channel;
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

