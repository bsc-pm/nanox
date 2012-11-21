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

#include "mpi.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "mpidd.hpp"
#include "mpiprocessor.hpp"
#include "workdescriptor_decl.hpp"
#include "processingelement_fwd.hpp"
#include "copydescriptor_decl.hpp"
#include "mpidevice.hpp"


//TODO: check for errors in communications
//TODO: Depending on multithread level, wait for answers or not
using namespace nanos;


MPI_Datatype MPIDevice::cacheStruct;

MPIDevice::MPIDevice(const char *n) : Device(n) {
}

/*! \brief MPIDevice copy constructor
 */
MPIDevice::MPIDevice(const MPIDevice &arch) : Device(arch) {
}

/*! \brief MPIDevice destructor
 */
MPIDevice::~MPIDevice() {
};

/* \breif allocate size bytes in the device
 */
void * MPIDevice::allocate(size_t size, ProcessingElement *pe) {
    std::cerr << "Inicio allocate\n";
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_ALLOCATE;
    order.hostAddr = 0;
    order.size = size;
    MPI_Status status;
    nanos::ext::MPIProcessor::nanos_MPI_Send(&order, 1, cacheStruct, myPE->_rank, TAG_CACHE_ORDER, myPE->_communicator);
    nanos::ext::MPIProcessor::nanos_MPI_Recv(&order, 1, cacheStruct, myPE->_rank, TAG_CACHE_ANSWER_ALLOC, myPE->_communicator, &status);
    std::cerr << "Fin allocate\n";
    return (void *) order.devAddr;
}

/* \brief free address
 */
void MPIDevice::free(void *address, ProcessingElement *pe) {
    std::cerr << "Inicio free\n";
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_FREE;
    order.devAddr = (uint64_t) address;
    order.size = 0;
    short ans;
    MPI_Status status;    
    nanos::ext::MPIProcessor::nanos_MPI_Send(&order, 1, cacheStruct, myPE->_rank, TAG_CACHE_ORDER, myPE->_communicator);
    //nanos::ext::MPIProcessor::nanos_MPI_Recv(&ans, 1, MPI_SHORT, myPE->_rank, TAG_CACHE_ANSWER_FREE, myPE->_communicator, &status);

    std::cerr << "Fin free\n";
}

/* \brief Reallocate and copy from address.
 */
void * MPIDevice::realloc(void *address, size_t size, size_t old_size, ProcessingElement *pe) {
    std::cerr << "Inicio REALLOC\n";
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_REALLOC;
    order.devAddr  = (uint64_t) address;
    order.hostAddr = 0;
    order.size = size;
    order.old_size = old_size;
    MPI_Status status;
    nanos::ext::MPIProcessor::nanos_MPI_Send(&order, 1, cacheStruct, myPE->_rank, TAG_CACHE_ORDER, myPE->_communicator);
    nanos::ext::MPIProcessor::nanos_MPI_Recv(&order, 1, cacheStruct, myPE->_rank, TAG_CACHE_ANSWER_REALLOC, myPE->_communicator, &status);

    std::cerr << "Fin realloc\n";
    printf("Copiado de %p\n",(void *) order.devAddr);
    return (void *) order.devAddr;
}

/* \brief Copy from remoteSrc in the host to localDst in the device
 *        Returns true if the operation is synchronous
 */
bool MPIDevice::copyIn(void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe) {
    std::cerr << "Inicio copyin\n";
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_COPYIN;
    order.devAddr = (uint64_t) localDst;
    order.hostAddr = (uint64_t) remoteSrc.getTag();
    order.size = size;
    short ans;
    MPI_Status status;
    printf("Dir copyin host%p a device %p, size %d\n",(void*) order.hostAddr,(void*) order.devAddr,order.size);
    nanos::ext::MPIProcessor::nanos_MPI_Send(&order, 1, cacheStruct, myPE->_rank, TAG_CACHE_ORDER, myPE->_communicator);
    nanos::ext::MPIProcessor::nanos_MPI_Ssend((void*) order.hostAddr, order.size, MPI_BYTE, myPE->_rank, TAG_CACHE_DATA_IN, myPE->_communicator);
    //nanos::ext::MPIProcessor::nanos_MPI_Recv(&ans, 1, MPI_SHORT, myPE->_rank, TAG_CACHE_ANSWER_CIN, myPE->_communicator, &status);
    std::cerr << "Fin copyin\n";
    return true;
}

/* \brief Copy from localSrc in the device to remoteDst in the host
 *        Returns true if the operation is synchronous
 */
bool MPIDevice::copyOut(CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe) {
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_COPYOUT;
    order.devAddr = (uint64_t) localSrc;
    order.hostAddr = (uint64_t) remoteDst.getTag();    
    order.size = size;
    printf("Inicio copyout host %p %d\n",(void*) order.hostAddr, order.size);
    MPI_Status status;
    nanos::ext::MPIProcessor::nanos_MPI_Send(&order, 1, cacheStruct, myPE->_rank, TAG_CACHE_ORDER, myPE->_communicator);
    nanos::ext::MPIProcessor::nanos_MPI_Recv((void*) order.hostAddr, order.size, MPI_BYTE, myPE->_rank, TAG_CACHE_DATA_OUT, myPE->_communicator, &status);
    //short ans;
    //nanos::ext::MPIProcessor::nanos_MPI_Recv(&ans, 1, MPI_SHORT, myPE->_rank, TAG_CACHE_ANSWER_COUT, myPE->_communicator, &status);
    std::cerr << "Fin copyout\n";
    return true;
}

/* \brief Copy localy in the device from src to dst
 */
void MPIDevice::copyLocal(void *dst, void *src, size_t size, ProcessingElement *pe) {
    std::cerr << "Inicio copylocal\n";
    //HostAddr will be src and DevAddr will be dst (both are device addresses)
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_COPYLOCAL;
    order.devAddr = (uint64_t) dst;
    order.hostAddr = (uint64_t) src;
    order.size = size;
    nanos::ext::MPIProcessor::nanos_MPI_Ssend(&order, 1, cacheStruct, myPE->_rank, TAG_CACHE_ORDER, myPE->_communicator);
    std::cerr << "Fin copyin\n";
}

void MPIDevice::syncTransfer(uint64_t hostAddress, ProcessingElement *pe) {
}

bool MPIDevice::copyDevToDev(void * addrDst, CopyDescriptor& cdDst, void * addrSrc, std::size_t size, ProcessingElement *peDst, ProcessingElement *peSrc) {
    return true;
}

void MPIDevice::initMPICacheStruct() {
    //Initialize cacheStruct in case it's not initialized
    if (cacheStruct == NULL) {
        MPI_Datatype typelist[5] = {MPI_SHORT, MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG};
        int blocklen[5] = {1, 1, 1, 1,1};
        MPI_Aint disp[5] = {offsetof(cacheOrder, opId), offsetof(cacheOrder, hostAddr), offsetof(cacheOrder, devAddr), offsetof(cacheOrder, size),offsetof(cacheOrder, old_size)};
        MPI_Type_create_struct(5, blocklen, disp, typelist, &cacheStruct);
        MPI_Type_commit(&cacheStruct);
    }
}

void MPIDevice::mpiCacheWorker() {
    //myThread = myThread->getNextThread();
    MPI_Comm parentcomm; /* intercommunicator */
    MPI_Comm_get_parent(&parentcomm);
    //If this process was not spawned, we don't need this daemon-thread
    if (parentcomm != NULL && parentcomm != MPI_COMM_NULL) {
        MPI_Status status;
        short ans=1;
        cacheOrder order;                    
        for (;;) {
            //if (!t->isRunning()) break; //{ std::cerr << "FINISHING MPI THD!" << std::endl; break; }
            //if (sys.getNetwork()->getNodeNum() == 0  ) std::cerr <<"ppp poll " << myThread->getId() << std::endl;
            //MPI_Comm_get_parent(&parentcomm);
            nanos::ext::MPIProcessor::nanos_MPI_Recv(&order, 1, cacheStruct, 0, TAG_CACHE_ORDER, parentcomm, &status);

            //TODO: make this operation single-message
            //and check performance with probe+remake struct datatype+single-message vs dual message
            switch (order.opId) {
                case OPID_FINISH:         
                    //MPI_Finalize();
                    pthread_exit ( 0 );
                    return;
                case OPID_COPYIN:
                {
                    std::cerr << "Hago un CopyIn en device\n";
                    printf("Dir copyin%p, size %d\n",(void*) order.devAddr,order.size);
                    //int incoming_msg_size;
                    //MPI_Probe(MPI_ANY_SOURCE, 98, parentcomm, &status);
                    //MPI_Get_count(&status, MPI_BYTE, &incoming_msg_size);
                    //MPI_Comm_get_parent(&parentcomm);
                    nanos::ext::MPIProcessor::nanos_MPI_Recv((void*) order.devAddr, order.size, MPI_BYTE, 0, TAG_CACHE_DATA_IN, parentcomm, &status);
                    float* arr= (float*) order.devAddr;
                    printf("Copio valor %f\n",arr[2047]);

                    //nanoxRegisterMPICopyIn(order.devAddr, order.hostAddr, order.size, order.opId);
                    //nanos::ext::MPIProcessor::nanos_MPI_Send(&ans, 1, MPI_SHORT, 0, TAG_CACHE_ANSWER_CIN, parentcomm);

                    std::cerr << "Fin un CopyIn en device\n";
                    break;
                }
                case OPID_COPYOUT:
                    printf("Hago un copyOut de device %p\n",(void *) order.devAddr);
                    //MPI_Comm_get_parent(&parentcomm);
                    nanos::ext::MPIProcessor::nanos_MPI_Send((void *) order.devAddr, order.size, MPI_BYTE, 0, TAG_CACHE_DATA_OUT, parentcomm);
                    //nanos::ext::MPIProcessor::nanos_MPI_Send(&ans, 1, MPI_SHORT, 0, TAG_CACHE_ANSWER_COUT, parentcomm);
                    std::cerr << "Fin copyOut en device\n";
                    break;
                case OPID_FREE:
                    std::cerr << "Hago un free en device\n";
                    //nanoxRegisterMPIFree((void*)order.devAddr, order.size);
                    delete[] (char *) order.devAddr;
                    printf("Dir free%p\n",(char*) order.devAddr);
                    //nanos::ext::MPIProcessor::nanos_MPI_Send(&ans, 1, MPI_SHORT, 0, TAG_CACHE_ANSWER_FREE, parentcomm);

                    std::cerr << "Fin free en device\n";
                    break;
                case OPID_ALLOCATE:           
                {
                    std::cerr << "Hago un allocate en device\n";
                    char* ptr = new char[order.size];
                    order.devAddr = (uint64_t) ptr;
                    printf("Dir alloc%p size %d\n",(void*) order.devAddr,order.size);
                    //MPI_Comm_get_parent(&parentcomm);
                    nanos::ext::MPIProcessor::nanos_MPI_Send(&order, 1, cacheStruct, 0, TAG_CACHE_ANSWER_ALLOC, parentcomm);
                    std::cerr << "Fin allocate en device\n";
                    break;
                }
                case OPID_REALLOC:           
                {
                    std::cerr << "Hago un reallocate en device\n";
                    delete[] (char *) order.devAddr;
                    char* ptr = new char[order.size];
                    printf("Realloc %d, %d\n", order.size,order.old_size);
                    printf("Copio de %p a %p, tam %d\n",(void*)  order.devAddr, ptr, order.old_size);
                    printf("Copio valor %f\n",((void*)  order.devAddr));
                    memcpy(ptr, (void*)  order.devAddr, order.old_size);
                    order.devAddr = (uint64_t) ptr;
                    //MPI_Comm_get_parent(&parentcomm);
                    nanos::ext::MPIProcessor::nanos_MPI_Send(&order, 1, cacheStruct, 0, TAG_CACHE_ANSWER_REALLOC, parentcomm);
                    std::cerr << "Fin reallocate en device\n";
                    break;
                }
                case OPID_COPYLOCAL:
                    std::cerr << "Hago un copylocal en device\n";
                    memcpy((void*) order.devAddr, (void*) order.hostAddr, order.size);
                    //nanos::ext::MPIProcessor::nanos_MPI_Send(&ans, 1, MPI_SHORT, 0, TAG_CACHE_ANSWER, parentcomm);
                    std::cerr << "Fin un copylocal en device\n";
                    break;
                default:
                    fatal("Received unknown operation id on MPI cache daemon thread");
            }
            //if ( sys.getNetwork()->getNodeNum() == 0 ) std::cerr <<"ppp poll complete " << myThread->getId() << std::endl;
            //if ( sys.getNetwork()->getNodeNum() == 0 ) std::cerr << "thread change! " << std::endl;
            //myThread = myThread->getNextThread();
        }
    }
}