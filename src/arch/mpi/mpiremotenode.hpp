/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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

using namespace nanos;
using namespace ext;
    

size_t MPIRemoteNode::_bufferDefaultSize = 0;
char* MPIRemoteNode::_bufferPtr = 0;
Lock MPIRemoteNode::_taskLock;
Lock MPIRemoteNode::_queueLock;
std::list<int> MPIRemoteNode::_pendingTasksQueue;
std::list<int> MPIRemoteNode::_pendingTaskParentsQueue;
bool MPIRemoteNode::_initialized=false;
int MPIRemoteNode::_currentTaskParent=-1;
int MPIRemoteNode::_currProcessor=0;

Lock& MPIRemoteNode::getTaskLock() {
    return _taskLock;
}

int MPIRemoteNode::getQueueCurrTaskIdentifier() {
    return _pendingTasksQueue.front();
}

int MPIRemoteNode::getQueueCurrentTaskParent() {
    return _pendingTaskParentsQueue.front();
}

int MPIRemoteNode::getCurrentTaskParent() {
    return _currentTaskParent;
}

void MPIRemoteNode::setCurrentTaskParent(int parent) {
    _currentTaskParent=parent;
}

int MPIRemoteNode::getCurrentProcessor() {
    return _currProcessor++;
}

void MPIRemoteNode::testTaskQueueSizeAndLock() {    
    _queueLock.acquire();    
    if (_pendingTasksQueue.size()==0) {
      _queueLock.release();
       getTaskLock().acquire();
    } else {
      _queueLock.release();
    }
}

void MPIRemoteNode::addTaskToQueue(int task_id, int parentId) {
    _queueLock.acquire();
    _pendingTasksQueue.push_back(task_id);
    _pendingTaskParentsQueue.push_back(parentId);
    _queueLock.release();
}

void MPIRemoteNode::removeTaskFromQueue() {
    _queueLock.acquire();
    _pendingTasksQueue.pop_front();
    _pendingTaskParentsQueue.pop_front();
    _queueLock.release();
}

////////////////////////////////
//Auxiliar filelock routines////
////////////////////////////////
//Lock not used with OMPI, remove unused function warning
#ifndef OPEN_MPI
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
    fcntl(fd, F_SETLKW, &lock);  /* Overwrites lock structure with preventors. */
    //remove( lockName );
    close( fd );
}
#endif

#endif
