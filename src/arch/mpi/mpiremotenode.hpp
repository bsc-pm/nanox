/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#ifndef _NANOS_MPI_REMOTE_NODE
#define _NANOS_MPI_REMOTE_NODE

#include "mpi.h"
#include "atomic_decl.hpp"
#include "mpiremotenode_decl.hpp"
#include "config.hpp"
#include "mpidevice.hpp"
#include "mpithread.hpp"
#include "cachedaccelerator.hpp"
#include "copydescriptor_decl.hpp"
#include "processingelement.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

namespace nanos {
namespace ext {
    
ProducerConsumerQueue<std::pair<int,int> >* MPIRemoteNode::_pendingTasksWithParent = NULL;
mpi::command::Dispatcher* MPIRemoteNode::_commandDispatcher = NULL;
std::vector<MPI_Datatype*> MPIRemoteNode::_taskStructsCache;
bool MPIRemoteNode::_initialized=false;
bool MPIRemoteNode::_disconnectedFromParent=false;
int MPIRemoteNode::_currentTaskParent=-1;
int MPIRemoteNode::_currProcessor=0;

mpi::command::Dispatcher& MPIRemoteNode::getDispatcher() {
    return *MPIRemoteNode::_commandDispatcher;
}

int MPIRemoteNode::getCurrentTaskParent() {
    return MPIRemoteNode::_currentTaskParent;
}

void MPIRemoteNode::setCurrentTaskParent(int parent) {
    MPIRemoteNode::_currentTaskParent=parent;
}

int MPIRemoteNode::getCurrentProcessor() {
    return MPIRemoteNode::_currProcessor++;
}

void MPIRemoteNode::addTaskToQueue(int task_id, int parent_id) {
    MPIRemoteNode::_pendingTasksWithParent->push( std::make_pair(task_id,parent_id) );
}

std::pair<int,int> MPIRemoteNode::getNextTaskAndParent() {
    return MPIRemoteNode::_pendingTasksWithParent->pop();
}

bool MPIRemoteNode::getDisconnectedFromParent(){
    return MPIRemoteNode::_disconnectedFromParent;
}

////////////////////////////////
//Auxiliar filelock routines////
////////////////////////////////
/*! Try to get lock. Return its file descriptor or -1 if failed.
 *
 *  @param lockName Name of file used as lock (i.e. '/var/lock/myLock').
 *  @return File descriptor of lock file, or -1 if failed.
 */
static int tryGetLock( char const *lockName )
{
    int fd = open( lockName, O_RDWR|O_CREAT, 0666 );   
    struct flock lock;
    lock.l_type    = F_WRLCK;   /* Test for any lock on any part of file. */
    lock.l_start   = 0;
    lock.l_whence  = SEEK_SET;
    lock.l_len     = 0;        
    if ( fcntl(fd, F_SETLKW, &lock) < 0)  {  /* Overwrites lock structure with preventors. */
        fd=-1;        
        close( fd );
    }
    return fd;
}

/*! Release the lock obtained with tryGetLock( lockName ).
 *
 *  @param fd File descriptor of lock returned by tryGetLock( lockName ).
 *  @param lockName Name of file used as lock (i.e. '/var/lock/myLock').
 */
static void releaseLock( int fd, char const *lockName )
{
    if( fd < 0 )
        return;
    struct flock lock;
    lock.l_type    = F_WRLCK;   /* Test for any lock on any part of file. */
    lock.l_start   = 0;
    lock.l_whence  = SEEK_SET;
    lock.l_len     = 0;        
    fcntl(fd, F_UNLCK, &lock);  /* Overwrites lock structure with preventors. */
    //remove( lockName );
    close( fd );
}

} // namespace mpi
} // namespace nanos

#endif
