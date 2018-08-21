/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

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

