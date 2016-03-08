
#ifndef INIT_HPP
#define INIT_HPP

#include "command.hpp"
#include "createauxthread.hpp"

namespace nanos {
namespace mpi {
namespace command {

typedef Command<OPID_TASK_INIT> Init;

/**
 * FIXME: Currently, GenericCommand does not have any member
 *        that can hold the task id that has to be executed.
 *
 * Executes a task given its identification number.
 */
template<>
void Init::Servant::serve()
{
	if( CreateAuxiliaryThread::Servant::isCreated() ) {
		//Add task to execution queue
		MPIRemoteNode::addTaskToQueue( _data.getCode(), _channel.getSource() );
	} else {
		//Execute the task in current thread
		MPIRemoteNode::setCurrentTaskParent( _channel.getSource() );
		MPIRemoteNode::executeTask( _data.getCode() );
	}
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // INIT_HPP

