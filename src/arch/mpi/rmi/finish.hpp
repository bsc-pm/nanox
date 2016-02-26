
#ifndef FINISH_HPP
#define FINISH_HPP

#include "command.hpp"
#include "mpiremotenode.hpp"

namespace nanos {
namespace mpi {
namespace command {

typedef Command<OPID_FINISH> Finish;

template<>
void Finish::Servant::serve()
{
	if(_createdExtraWorkerThread) {
		//Add finish task to execution queue
		MPIRemoteNode::addTaskToQueue(TASK_END_PROCESS,parentRank);
		MPIRemoteNode::getTaskLock().acquire();
		MPIRemoteNode::getTaskLock().acquire();
	} else {
		//Execute in current thread
		MPIRemoteNode::setCurrentTaskParent(parentRank);
		MPIRemoteNode::executeTask(TASK_END_PROCESS);
	}
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // FINISH_HPP
