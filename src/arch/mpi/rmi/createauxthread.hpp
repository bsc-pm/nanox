
#ifndef CREATE_AUX_THREAD_HPP
#define CREATE_AUX_THREAD_HPP

#include "atomic.hpp"
#include "command.hpp"
#include "commandchannel.hpp"

#include "mpidevice_decl.hpp" // OPID definitions

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
		CreateAuxiliartyThread::payload_type      _data;
		CreateAuxiliartyThread::main_channel_type _channel;

		static Atomic<bool>                       _alreadyCreated;

	public:
		CommandServant( Channel const& channel ) :
			_data( command_id ),
			_channel( channel )
		{
		}

		CommandServant( Channel const& channel, Payload const& data ) :
			_data( data ), _channel( channel )
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

		virtual void serve();
};

template<>
void CreateAuxiliaryThread::Requestor::dispatch()
{
}

/**
 * Creates a cache thread that manages masters' communications
 *
 * The thread is created only once. However, multiple request
 * may come from different masters.
 */
template<>
void CreateAuxiliaryThread::Servant::serve()
{
	bool created = _alreadyCreated. 
	if( !_alreadyCreated ) {
		_alreadyCreated = true;

		MPIDevice::createExtraCacheThread();
		MPIRemoteNode::nanosMPIWorker();
	}
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // CREATE_AUX_THREAD

