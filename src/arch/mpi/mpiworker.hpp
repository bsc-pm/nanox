
#ifndef MPI_WORKER_HPP
#define MPI_WORKER_HPP

#include "mpidevice_decl.hpp"
#include "mpiremotenode_decl.hpp"
#include "commanddispatcher.hpp"

#include "createauxthread.hpp"
#include "finish.hpp"

#include <mpi.h>

namespace nanos {

// Dedicated cache worker
template<>
void MPIDevice::remoteNodeCacheWorker<true>() {

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
void MPIDevice::remoteNodeCacheWorker<false>() {

    mpi::command::Dispatcher& dispatcher = MPIRemoteNode::getDispatcher();
    dispatcher.waitForCommands();
    dispatcher.queueAvailableCommands();
    dispatcher.executeCommands();
}

void MPIDevice::createExtraCacheThread() {
    //Create extra worker thread
    MPI_Comm mworld= MPI_COMM_WORLD;
    ext::SMPProcessor *core = sys.getSMPPlugin()->getLastFreeSMPProcessorAndReserve();
    if (core==NULL) {
        core = sys.getSMPPlugin()->getSMPProcessorByNUMAnode(0,MPIRemoteNode::getCurrentProcessor());
    }
    MPIProcessor *mpi = NEW MPIProcessor(&mworld, CACHETHREADRANK,-1, false, false, /* Dummy*/ MPI_COMM_SELF, core, /* Dummmy memspace */ 0);
    MPIDD * dd = NEW MPIDD((MPIDD::work_fct) MPIDevice::remoteNodeCacheWorker<true> );
    WD* wd = NEW WD(dd);
    NANOS_INSTRUMENT( sys.getInstrumentation()->incrementMaxThreads(); )
    mpi->startMPIThread(wd);
}

namespace ext {

int MPIRemoteNode::nanosMPIWorker() {
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

void MPIRemoteNode::mpiOffloadSlaveMain() {
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

