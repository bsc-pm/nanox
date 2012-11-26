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

#include "mpiprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "config.hpp"
#include "mpithread.hpp"
#include <iostream>
using namespace nanos;
using namespace nanos::ext;

System::CachePolicyType MPIProcessor::_cachePolicy = System::WRITE_THROUGH;
size_t MPIProcessor::_cacheDefaultSize = 10485800;
size_t MPIProcessor::_bufferDefaultSize = 0;
char* MPIProcessor::_bufferPtr = 0;
std::string MPIProcessor::_mpiFilename;
std::string MPIProcessor::_mpiFileArgs;
std::string MPIProcessor::_mpiHosts;
std::string MPIProcessor::_mpiMachinefile;

MPIProcessor::MPIProcessor(int id, MPI_Comm communicator, int rank) : CachedAccelerator<MPIDevice>(id, &MPI) {
    _communicator = communicator;
    _rank = rank;
}

void MPIProcessor::prepareConfig(Config &config) {

    config.registerConfigOption("mpi-exec", NEW Config::StringVar(_mpiFileArgs), "Defines secondary mpi file spawned in DEEP_Booster_Alloc");
    config.registerArgOption("mpi-exec", "mpi-exec");
    config.registerEnvOption("mpi-exec", "NX_MPIEXEC");

    config.registerConfigOption("mpimachinefile", NEW Config::StringVar(_mpiMachinefile), "Defines hosts file where secondary process can spawn in DEEP_Booster_Alloc\nThe format of the file is one host per line with blank lines and lines beginning with # ignored\nMultiple processes per host can be specified by specifying the host name as follows: hostA:n");
    config.registerArgOption("mpimachinefile", "mpimachinefile");
    config.registerEnvOption("mpimachinefile", "NX_MPIMACHINEFILE");

    config.registerConfigOption("mpihosts", NEW Config::StringVar(_mpiHosts), "Defines hosts file where secondary process can spawn in DEEP_Booster_Alloc\nExample: hostA hostB:2 hostC");
    config.registerArgOption("mpihosts", "mpihosts");
    config.registerEnvOption("mpihosts", "NX_MPIHOSTS");


    // Set the cache policy for MPI devices
    System::CachePolicyConfig *cachePolicyCfg = NEW System::CachePolicyConfig(_cachePolicy);
    cachePolicyCfg->addOption("wt", System::WRITE_THROUGH);
    cachePolicyCfg->addOption("wb", System::WRITE_BACK);
    cachePolicyCfg->addOption("nocache", System::NONE);
    config.registerConfigOption("mpi-cache-policy", cachePolicyCfg, "Defines the cache policy for MPI architectures: write-through / write-back (wb by default)");
    config.registerEnvOption("mpi-cache-policy", "NX_MPI_CACHE_POLICY");
    config.registerArgOption("mpi-cache-policy", "mpi-cache-policy");


    config.registerConfigOption("mpi-cache-size", NEW Config::SizeVar(_cacheDefaultSize), "Defines size of the cache for MPI allocated devices");
    config.registerArgOption("mpi-cache-size", "mpi-cache-size");
    config.registerEnvOption("mpi-cache-size", "NX_MPICACHESIZE");


    config.registerConfigOption("mpi-buffer-size", NEW Config::SizeVar(_bufferDefaultSize), "Defines size of the nanox MPI Buffer (MPI_Buffer_Attach/detach)");
    config.registerArgOption("mpi-buffer-size", "mpi-buffer-size");
    config.registerEnvOption("mpi-buffer-size", "NX_MPIBUFFERSIZE");
}

WorkDescriptor & MPIProcessor::getWorkerWD() const {
    MPIDD * dd = NEW MPIDD((MPIDD::work_fct)Scheduler::workerLoop);
    WD *wd = NEW WD(dd);
    return *wd;
}

WorkDescriptor & MPIProcessor::getMasterWD() const {
    WD * wd = NEW WD(NEW MPIDD());
    return *wd;
}

BaseThread &MPIProcessor::createThread(WorkDescriptor &helper) {
    MPIThread &th = *NEW MPIThread(helper, this);

    return th;
}

void MPIProcessor::setMpiFilename(char* new_name) {
    std::string tmp = std::string(new_name);
    _mpiFilename = tmp;
}

std::string MPIProcessor::getMpiFilename() {
    return _mpiFilename;
}

void MPIProcessor::DEEP_Booster_free(MPI_Comm *intercomm, int rank) {
    if (_bufferDefaultSize != 0 && _bufferPtr != 0) {
        int size;
        void *ptr;
        MPI_Buffer_detach(&ptr, &size);
        if (ptr != _bufferPtr) {
            warning("Another MPI Buffer was attached instead of the one defined with"
                    " nanox mpi buffer size, not releasing it");
            MPI_Buffer_attach(ptr, size);
        } else {
            MPI_Buffer_detach(&ptr, &size);
        }
        delete[] _bufferPtr;
    }
    cacheOrder order;
    order.opId = -1;
    int id = -1;
    if (rank == -1) {
        int size;
        MPI_Comm_remote_size(*intercomm, &size);
        for (int i = 0; i < size; i++) {
            //Closing cache daemon and user-level daemon
            nanos_MPI_Send(&id, 1, MPI_INT, i, TAG_INI_TASK, *intercomm);
            nanos_MPI_Send(&order, 1, nanos::MPIDevice::cacheStruct, i, TAG_CACHE_ORDER, *intercomm);
        }
    } else {
        nanos_MPI_Send(&order, 1, nanos::MPIDevice::cacheStruct, rank, TAG_CACHE_ORDER, *intercomm);
        nanos_MPI_Send(&id, 1, MPI_INT, rank, TAG_INI_TASK, *intercomm);
    }
}

/**
 * All this tasks redefine nanox messages
 */
void MPIProcessor::nanos_MPI_Init(int *argc, char ***argv) {
    int provided, claimed;
    //TODO: Try with multiple MPI thread
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Query_thread(&claimed);
    if (claimed < MPI_THREAD_SERIALIZED) {
        fatal0("MPI_Query_Thread returned multithread support less than MPI_THREAD_SERIALIZED, check your MPI "
                "implementation and try to configure it so it can support higher multithread levels");
    }
    if (_bufferDefaultSize != 0 && _bufferPtr != 0) {
        _bufferPtr = new char[_bufferDefaultSize];
        MPI_Buffer_attach(_bufferPtr, _bufferDefaultSize);
    }
    nanos::MPIDevice::initMPICacheStruct();
    MPI_Comm parentcomm; /* intercommunicator */
    MPI_Comm_get_parent(&parentcomm);

    //If this process was not spawned, we don't need the daemon-thread
    if (parentcomm != NULL && parentcomm != MPI_COMM_NULL) {
        //Initialice MPI PE with a communicator and special rank for the cache thread
        PE *mpi = NEW nanos::ext::MPIProcessor(999, MPI_COMM_WORLD, CACHETHREADRANK);
        MPIDD * dd = NEW MPIDD((MPIDD::work_fct) nanos::MPIDevice::mpiCacheWorker);
        WD *wd = NEW WD(dd);
        &mpi->startThread(*wd);
    }
}

int MPIProcessor::nanos_MPI_Send_taskinit(void *buf, int count, MPI_Datatype datatype, int dest,
        MPI_Comm comm) {
    return nanos_MPI_Send(buf, count, datatype, dest, TAG_INI_TASK, comm);
}

int MPIProcessor::nanos_MPI_Recv_taskinit(void *buf, int count, MPI_Datatype datatype, int source,
        MPI_Comm comm, MPI_Status *status) {
    return nanos_MPI_Recv(buf, count, datatype, source, TAG_INI_TASK, comm, status);
}

int MPIProcessor::nanos_MPI_Send_taskend(void *buf, int count, MPI_Datatype datatype, int dest,
        MPI_Comm comm) {
    return nanos_MPI_Send(buf, count, datatype, dest, TAG_END_TASK, comm);
}

int MPIProcessor::nanos_MPI_Recv_taskend(void *buf, int count, MPI_Datatype datatype, int source,
        MPI_Comm comm, MPI_Status *status) {
    return nanos_MPI_Recv(buf, count, datatype, source, TAG_END_TASK, comm, status);
}

int MPIProcessor::nanos_MPI_Send_datastruct(void *buf, int count, MPI_Datatype datatype, int dest,
        MPI_Comm comm) {
    return nanos_MPI_Send(buf, count, datatype, dest, TAG_ENV_STRUCT, comm);
}

int MPIProcessor::nanos_MPI_Recv_datastruct(void *buf, int count, MPI_Datatype datatype, int source,
        MPI_Comm comm, MPI_Status *status) {
    return nanos_MPI_Recv(buf, count, datatype, source, TAG_ENV_STRUCT, comm, status);
}

int MPIProcessor::nanos_MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
        MPI_Comm comm) {
    if (dest==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        dest=myPE->_rank;
        comm=myPE->_communicator;
    }
    int err = MPI_Send(buf, count, datatype, dest, tag, comm);
    return err;
}

int MPIProcessor::nanos_MPI_Ssend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
        MPI_Comm comm) {
    if (dest==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        dest=myPE->_rank;
        comm=myPE->_communicator;
    }
    int err = MPI_Ssend(buf, count, datatype, dest, tag, comm);
    return err;
}

int MPIProcessor::nanos_MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
        MPI_Comm comm, MPI_Status *status) {
    if (source==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        source=myPE->_rank;
        comm=myPE->_communicator;
    }
    int err = MPI_Recv(buf, count, datatype, source, tag, comm, status);
    return err;
}