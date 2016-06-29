
#ifndef INIT_HPP
#define INIT_HPP

#include "command.hpp"
#include "createauxthread.hpp"

namespace nanos {
namespace mpi {
namespace command {

typedef Command<OPID_TASK_INIT> Init;

template<>
void Init::Requestor::dispatch()
{
}

/**
 * Executes a task given its identification number.
 */
template<>
void Init::Servant::serve()
{
	//Add task to execution queue
	MPIRemoteNode::addTaskToQueue( _data.getCode(), _channel.getSource() );
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // INIT_HPP

