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
std::string MPIProcessor::_mpiExecFile;
std::string MPIProcessor::_mpiHosts;
std::string MPIProcessor::_mpiHostsFile;
unsigned int* MPIProcessor::_mpiFileHashname;
unsigned int* MPIProcessor::_mpiFileSize;
int MPIProcessor::_mpiFileArrSize;

MPIProcessor::MPIProcessor(int id, MPI_Comm communicator, int rank) : CachedAccelerator<MPIDevice>(id, &MPI) {
    _communicator = communicator;
    _rank = rank;
}

void MPIProcessor::prepareConfig(Config &config) {

    config.registerConfigOption("mpi-exec", NEW Config::StringVar(_mpiExecFile), "Defines secondary mpi file spawned in DEEP_Booster_Alloc");
    config.registerArgOption("mpi-exec", "mpi-exec");
    config.registerEnvOption("mpi-exec", "NX_MPIEXEC");

    config.registerConfigOption("mpihostsfile", NEW Config::StringVar(_mpiHostsFile), "Defines hosts file where secondary process can spawn in DEEP_Booster_Alloc\nThe format of the file is: One host per line with blank lines and lines beginning with # ignored\nMultiple processes per host can be specified by specifying the host name as follows: hostA:n");
    config.registerArgOption("mpihostsfile", "mpihostsfile");
    config.registerEnvOption("mpihostsfile", "NX_MPIHOSTSFILE");

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

void MPIProcessor::setMpiExename(char* new_name) {
    std::string tmp = std::string(new_name);
    _mpiFilename = tmp;
}

std::string MPIProcessor::getMpiExename() {
    return _mpiFilename;
}

void MPIProcessor::DEEP_Booster_free(MPI_Comm *intercomm, int rank) {
//    if (_bufferDefaultSize != 0 && _bufferPtr != 0) {
//        int size;
//        void *ptr;
//        MPI_Buffer_detach(&ptr, &size);
//        if (ptr != _bufferPtr) {
//            warning("Another MPI Buffer was attached instead of the one defined with"
//                    " nanox mpi buffer size, not releasing it");
//            MPI_Buffer_attach(ptr, size);
//        } else {
//            MPI_Buffer_detach(&ptr, &size);
//        }
//        delete[] _bufferPtr;
//    }
    cacheOrder order;
    order.opId = -1;
    int id = -1; 
    if (rank == -1) {
        int size;
        MPI_Comm_remote_size(*intercomm, &size);
        for (int i = 0; i < size; i++) {
            //Closing cache daemon and user-level daemon
            nanos_MPI_Send(&order, 1, nanos::MPIDevice::cacheStruct, i, TAG_CACHE_ORDER, *intercomm);
            nanos_MPI_Send(&id, 1, MPI_INT, i, TAG_INI_TASK, *intercomm);
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
    if (claimed < MPI_THREAD_MULTIPLE) {
        fatal0("MPI_Query_Thread returned multithread support less than MPI_THREAD_MULTIPLE, check your MPI "
                "implementation and try to configure it so it can support this multithread level");
    }
    if (_bufferDefaultSize != 0 && _bufferPtr != 0) {
        _bufferPtr = new char[_bufferDefaultSize];
        MPI_Buffer_attach(_bufferPtr, _bufferDefaultSize);
    }
    nanos::MPIDevice::initMPICacheStruct();

    //If this process was spawned, start the daemon-thread
    if ((*argc) > 1 && !strcmp((*argv)[(*argc) - 1], TAG_MAIN_OMPSS)){
         //In this case we are child, when nanox spawns us, it fills both args
        setMpiExename((*argv)[(*argc)-2]);
        //Initialice MPI PE with a communicator and special rank for the cache thread
        PE *mpi = NEW nanos::ext::MPIProcessor(999, MPI_COMM_WORLD, CACHETHREADRANK);
        MPIDD * dd = NEW MPIDD((MPIDD::work_fct) nanos::MPIDevice::mpiCacheWorker);
        WD *wd = NEW WD(dd);
        mpi->startThread(*wd);
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
    //printf("Envio con tag %d, a %d\n",tag,dest);
    int err = MPI_Send(buf, count, datatype, dest, tag, comm);
    //printf("Fin Envio con tag %d, a %d\n",tag,dest);
    return err;
}

int MPIProcessor::nanos_MPI_Ssend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
        MPI_Comm comm) {
    if (dest==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        dest=myPE->_rank;
        comm=myPE->_communicator;
    }
    //printf("Enviobloq con tag %d, a %d\n",tag,dest);
    int err = MPI_Ssend(buf, count, datatype, dest, tag, comm);
    //printf("Fin Enviobloq con tag %d, a %d\n",tag,dest);
    return err;
}

int MPIProcessor::nanos_MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
        MPI_Comm comm, MPI_Status *status) {
    if (source==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        source=myPE->_rank;
        comm=myPE->_communicator;
    }
    //printf("recv con tag %d, desde %d\n",tag,source);
    int err = MPI_Recv(buf, count, datatype, source, tag, comm, status);
    //printf("Fin recv con tag %d, desde %d\n",tag,source);
    return err;
}