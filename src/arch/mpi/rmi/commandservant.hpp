
#ifndef COMMAND_SERVANT_HPP
#define COMMAND_SERVANT_HPP

namespace nanos {
namespace mpi {
namespace command {

template< int command_id, typename Payload, typename Channel >
class CommandServant : public Channel {
	private:
		Payload _data;
	public:
		CommandServant() :
			Channel(),
			_data( command_id )
		{
		}

		virtual ~CommandServant()
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
		 * Perform action on the remote process
		 * To be defined by each specific operation
		 */
		void serve();
};

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COMMAND_SERVANT_HPP

