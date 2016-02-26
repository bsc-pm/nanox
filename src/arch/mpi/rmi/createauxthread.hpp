
#ifndef CREATE_AUX_THREAD_HPP
#define CREATE_AUX_THREAD_HPP

#include "command.hpp"

namespace nanos {
namespace mpi {
namespace command {

typedef Command<OPID_CREATEAUXTHREAD> CreateAuxiliaryThread;

template<>
void CreateAuxiliaryThread::Requestor::dispatch()
{
	// FIXME: this function currently does nothing. We might
	// however move the sendCommand to dispatch from the constructor
	// to avoid several messages being sent.
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
	if( !_createdExtraWorkedThread ) {
		_createdExtraWorkerThread = true;

		MPIDevice::createExtraCacheThread();
		MPIRemoteNode::nanosMPIWorker();
	}
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // CREATE_AUX_THREAD

