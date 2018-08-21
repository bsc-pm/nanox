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

#ifndef _NANOS_MPI_REMOTE_NODE
#define _NANOS_MPI_REMOTE_NODE

#include "mpiremotenode_decl.hpp"

#include "atomic_decl.hpp"
#include "config.hpp"
#include "cachedaccelerator.hpp"
#include "copydescriptor_decl.hpp"
#include "processingelement.hpp"

#include "mpidevice.hpp"
#include "mpithread.hpp"

#include "concurrent_queue.hpp"

#include <mpi.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <unistd.h>

namespace nanos {
namespace ext {

inline mpi::command::Dispatcher& MPIRemoteNode::getDispatcher() {
    return *MPIRemoteNode::_commandDispatcher;
}

inline int MPIRemoteNode::getCurrentTaskParent() {
    return MPIRemoteNode::_currentTaskParent;
}

inline void MPIRemoteNode::setCurrentTaskParent(int parent) {
    MPIRemoteNode::_currentTaskParent=parent;
}

inline int MPIRemoteNode::getCurrentProcessor() {
    return MPIRemoteNode::_currProcessor++;
}

inline bool MPIRemoteNode::isNextTaskAvailable() {
    return !MPIRemoteNode::_pendingTasksWithParent->empty();
}

inline void MPIRemoteNode::addTaskToQueue(int task_id, int parent_id) {
    MPIRemoteNode::_pendingTasksWithParent->push( std::make_pair(task_id,parent_id) );
}

inline std::pair<int,int> MPIRemoteNode::getNextTaskAndParent() {
    return MPIRemoteNode::_pendingTasksWithParent->pop();
}

inline bool MPIRemoteNode::getDisconnectedFromParent(){
    return MPIRemoteNode::_disconnectedFromParent;
}

inline void MPIRemoteNode::registerSpawn( MPI_Comm communicator, mpi::RemoteSpawn& spawn ) {
    ensure0( _spawnedRemotes.find( communicator ) == _spawnedRemotes.end(),
             "A remote group can only be inserted once (communicator already present)" );
    _spawnedRemotes.insert( std::make_pair( communicator, &spawn ) );
}

inline RemoteSpawnMap& MPIRemoteNode::getRegisteredSpawns() {
    return _spawnedRemotes;
}

} // namespace mpi
} // namespace nanos

#endif
