
#ifndef CREATE_AUX_THREAD_HPP
#define CREATE_AUX_THREAD_HPP

#include "atomic_flag.hpp"
#include "command.hpp"
#include "commandchannel.hpp"

#include "mpidevice.hpp"

namespace nanos {
namespace mpi {
namespace command {

typedef Command<OPID_CREATEAUXTHREAD> CreateAuxiliaryThread;

template<>
class CommandServant<
    CreateAuxiliaryThread::id,
    CreateAuxiliaryThread::payload_type,
    CreateAuxiliaryThread::main_channel_type
                      > : public BaseServant {
	private:
		typedef CreateAuxiliaryThread::payload_type payload_type;
		typedef CreateAuxiliaryThread::main_channel_type main_channel_type;

		payload_type       _data;
		main_channel_type  _channel;

		static atomic_flag _alreadyCreated;

	public:
		CommandServant( const main_channel_type& channel ) :
			_data(), _channel( channel )
		{
			_data.initialize( CreateAuxiliaryThread::id );
		}

		CommandServant( const main_channel_type& channel, const payload_type& data ) :
			_data( data ), _channel( channel )
		{
		}

		virtual ~CommandServant()
		{
		}

		payload_type &getData()
		{
			return _data;
		}

		payload_type const& getData() const
		{
			return _data;
		}

		virtual void serve();

		static bool isCreated()
		{
			return _alreadyCreated.load();
		}
};

atomic_flag CreateAuxiliaryThread::Servant::_alreadyCreated;

template<>
inline void CreateAuxiliaryThread::Requestor::dispatch()
{
}

/**
 * Creates a cache thread that manages masters' communications
 *
 * The thread is created only once. However, multiple request
 * may come from different masters.
 */
inline void CreateAuxiliaryThread::Servant::serve()
{
	bool alreadyCreated = _alreadyCreated.test_and_set();
	if( !alreadyCreated ) {
		MPIDevice::createExtraCacheThread();
	}
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // CREATE_AUX_THREAD

