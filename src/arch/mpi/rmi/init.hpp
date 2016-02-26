
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
void Init::Servant::serve()
{
	// FIXME: Currently, GenericCommand does not have any member
	// which can store a task id.
	int task_id=(int)order.hostAddr;

	// If our node is used by more than one parent
	// create the worker thread as we may have cache orders from
	// one node while executing tasks from the other node
	if ( firstParent!=parentRank && !_createdExtraWorkerThread ) {
		if (firstParent==-1) {
			firstParent=parentRank;
		} else {
			// Add current task to queue and become executor
			MPIRemoteNode::addTaskToQueue(task_id,parentRank);

			// Create extra thread which controls the cache
			CreateAuxiliaryThread::Servant order;
			order.serve();
			return;
		}
	}

	if (_createdExtraWorkerThread) {
		//Add task to execution queue
		MPIRemoteNode::addTaskToQueue(task_id,parentRank);
	} else {
		//Execute the task in current thread
		MPIRemoteNode::setCurrentTaskParent(parentRank);
		MPIRemoteNode::executeTask(task_id);
	}
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // INIT_HPP

