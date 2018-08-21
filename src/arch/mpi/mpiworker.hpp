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

#ifndef MPI_WORKER_HPP
#define MPI_WORKER_HPP

#include "mpidevice.hpp"
#include "mpiremotenode_decl.hpp"
#include "commanddispatcher.hpp"

#include "createauxthread.hpp"
#include "finish.hpp"

#include <mpi.h>

namespace nanos {

using namespace nanos::ext;

// Dedicated cache worker
template<>
inline void MPIDevice::remoteNodeCacheWorker<true>() {

    mpi::command::Dispatcher& dispatcher = MPIRemoteNode::getDispatcher();
    // Before waiting for new commands, execute those that were left
    // pending by the regular mpi slave thread
    dispatcher.executeCommands();
    while( !mpi::command::Finish::Servant::isFinished() ) {
        dispatcher.waitForCommands();
        dispatcher.queueAvailableCommands();
        dispatcher.executeCommands();
    }
}

// Worker thread that waits for commands as well
template<>
inline void MPIDevice::remoteNodeCacheWorker<false>() {

    mpi::command::Dispatcher& dispatcher = MPIRemoteNode::getDispatcher();
    dispatcher.waitForCommands();
    dispatcher.queueAvailableCommands();
    dispatcher.executeCommands();
}

inline void MPIDevice::createExtraCacheThread() {
    //Create extra worker thread
    SMPProcessor *core = sys.getSMPPlugin()->getLastFreeSMPProcessorAndReserve();
    if (core==NULL) {
        core = sys.getSMPPlugin()->getSMPProcessorByNUMAnode( 0, MPIRemoteNode::getCurrentProcessor() );
    }
    MPIProcessor *mpi = NEW MPIProcessor( MPI_COMM_WORLD, CACHETHREADRANK, false, /* Dummy*/ MPI_COMM_SELF, core, /* Dummmy memspace */ 0);
    MPIDD * dd = NEW MPIDD((MPIDD::work_fct) MPIDevice::remoteNodeCacheWorker<true> );
    WD* wd = NEW WD(dd);
    mpi->startMPIThread(wd);
}

namespace ext {

inline int MPIRemoteNode::nanosMPIWorker() {
    // Meanwhile communications thread is not created
    while( !mpi::command::CreateAuxiliaryThread::Servant::isCreated()
           && !mpi::command::Finish::Servant::isFinished() ) {

        nanos::MPIDevice::remoteNodeCacheWorker<false>();

        if( MPIRemoteNode::isNextTaskAvailable() ) {
            std::pair<int,int> taskWithParent = MPIRemoteNode::getNextTaskAndParent();
            setCurrentTaskParent( taskWithParent.second );
            executeTask( taskWithParent.first );
        }
    }

    // When cache thread has already been created
    // just take pending offload tasks created by communications thread
    while( !mpi::command::Finish::Servant::isFinished() ) {
        std::pair<int,int> taskWithParent = MPIRemoteNode::getNextTaskAndParent();

        setCurrentTaskParent( taskWithParent.second );
        executeTask( taskWithParent.first );
    }
    return 0;
}

inline void MPIRemoteNode::mpiOffloadSlaveMain() {
    MPI_Comm parentcomm; /* intercommunicator */
    MPI_Comm_get_parent(&parentcomm);    

    // create offload task queue
    MPIRemoteNode::_pendingTasksWithParent =
                                 new ProducerConsumerQueue<std::pair<int,int> >();
    // reserve space for 10 elements of each generic command type
    MPIRemoteNode::_commandDispatcher =
                                 new mpi::command::Dispatcher(parentcomm, 10 );

    nanosMPIWorker();
}

} // namespace ext
} // namespace nanos

#endif // MPI_WORKER_HPP

