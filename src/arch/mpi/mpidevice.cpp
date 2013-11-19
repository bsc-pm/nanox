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
Directory* MPIDevice::_masterDir;
bool MPIDevice::executingTask=false;

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
void * MPIDevice::allocate(size_t size, ProcessingElement *pe, uint64_t tag ) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_ALLOC_EVENT);
    //std::cerr << "Inicio allocate\n";
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_ALLOCATE;
    order.hostAddr = 0;
    order.size = size;
    //MPI_Status status;
    nanos::ext::MPIProcessor::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_CACHE_ORDER, myPE->getCommunicator());
    nanos::ext::MPIProcessor::nanosMPIRecv(&order, 1, cacheStruct, myPE->getRank(), TAG_CACHE_ANSWER_ALLOC, myPE->getCommunicator(), MPI_STATUS_IGNORE );
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    //std::cerr << "Fin allocate\n";
    if (order.devAddr==0){
        return CACHE_ALLOC_ERROR;
    }
    return (void *) order.devAddr;
}

/* \brief free address
 */
void MPIDevice::free(void *address, ProcessingElement *pe) {
    //std::cerr << "Inicio free\n";
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_FREE_EVENT);
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_FREE;
    order.devAddr = (uint64_t) address;
    order.size = 0;
    //short ans;
    //MPI_Status status;    
    nanos::ext::MPIProcessor::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_CACHE_ORDER, myPE->getCommunicator());
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    //nanos::ext::MPIProcessor::nanos_MPI_Recv(&ans, 1, MPI_SHORT, myPE->getRank(), TAG_CACHE_ANSWER_FREE, myPE->_communicator, MPI_STATUS_IGNORE );

    //std::cerr << "Fin free\n";
}

/* \brief Reallocate and copy from address.
 */
void * MPIDevice::realloc(void *address, size_t size, size_t old_size, ProcessingElement *pe) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_REALLOC_EVENT);
    //std::cerr << "Inicio REALLOC\n";
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_REALLOC;
    order.devAddr  = (uint64_t) address;
    order.hostAddr = 0;
    order.size = size;
    //order.old_size = old_size;
    //MPI_Status status;
    nanos::ext::MPIProcessor::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_CACHE_ORDER, myPE->getCommunicator());
    nanos::ext::MPIProcessor::nanosMPIRecv(&order, 1, cacheStruct, myPE->getRank(), TAG_CACHE_ANSWER_REALLOC, myPE->getCommunicator(), MPI_STATUS_IGNORE );

    //std::cerr << "Fin realloc\n";
    //printf("Copiado de %p\n",(void *) order.devAddr);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return (void *) order.devAddr;
}

/* \brief Copy from remoteSrc in the host to localDst in the device
 *        Returns true if the operation is synchronous
 */
bool MPIDevice::copyIn(void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_COPYIN_SYNC_EVENT);
    //std::cerr << "Inicio copyin\n";
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_COPYIN;
    order.devAddr = (uint64_t) localDst;
    order.hostAddr = (uint64_t) remoteSrc.getTag();
    order.size = size;
    //short ans;
    //MPI_Status status;
    //printf("Dir copyin host%p a device %p, size %lu\n",(void*) order.hostAddr,(void*) order.devAddr,order.size);
    nanos::ext::MPIProcessor::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_CACHE_ORDER, myPE->getCommunicator());
    nanos::ext::MPIProcessor::nanosMPISsend((void*) order.hostAddr, order.size, MPI_BYTE, myPE->getRank(), TAG_CACHE_DATA_IN, myPE->getCommunicator());
    //nanos::ext::MPIProcessor::nanos_MPI_Recv(&ans, 1, MPI_SHORT, myPE->getRank(), TAG_CACHE_ANSWER_CIN, myPE->_communicator, MPI_STATUS_IGNORE );
    //std::cerr << "Fin copyin\n";
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return true;
}

/* \brief Copy from localSrc in the device to remoteDst in the host
 *        Returns true if the operation is synchronous
 */
bool MPIDevice::copyOut(CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_COPYOUT_SYNC_EVENT);
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_COPYOUT;
    order.devAddr = (uint64_t) localSrc;
    order.hostAddr = (uint64_t) remoteDst.getTag();    
    order.size = size;
    //printf("Inicio copyout host %p %lu\n",(void*) order.hostAddr, order.size);
    //MPI_Status status;
    nanos::ext::MPIProcessor::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_CACHE_ORDER, myPE->getCommunicator());
    nanos::ext::MPIProcessor::nanosMPIRecv((void*) order.hostAddr, order.size, MPI_BYTE, myPE->getRank(), TAG_CACHE_DATA_OUT, myPE->getCommunicator(), MPI_STATUS_IGNORE );
    //short ans;
    //nanos::ext::MPIProcessor::nanos_MPI_Recv(&ans, 1, MPI_SHORT, myPE->getRank(), TAG_CACHE_ANSWER_COUT, myPE->_communicator, MPI_STATUS_IGNORE );
    //std::cerr << "Fin copyout\n";
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return true;
}

/* \brief Copy localy in the device from src to dst
 */
void MPIDevice::copyLocal(void *dst, void *src, size_t size, ProcessingElement *pe) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_COPYLOCAL_SYNC_EVENT);
    //std::cerr << "Inicio copylocal\n";
    //HostAddr will be src and DevAddr will be dst (both are device addresses)
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    cacheOrder order;
    order.opId = OPID_COPYLOCAL;
    order.devAddr = (uint64_t) dst;
    order.hostAddr = (uint64_t) src;
    order.size = size;
    nanos::ext::MPIProcessor::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_CACHE_ORDER, myPE->getCommunicator());
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    //std::cerr << "Fin copyin\n";
}

void MPIDevice::syncTransfer(uint64_t hostAddress, ProcessingElement *pe) {
}

bool MPIDevice::copyDevToDev(void * addrDst, CopyDescriptor& cdDst, void * addrSrc, std::size_t size, ProcessingElement *peDst, ProcessingElement *peSrc) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_COPYDEV2DEV_SYNC_EVENT);
    //This will never be another PE type which is not MPI (or something is broken in core :) )
    nanos::ext::MPIProcessor * src = (nanos::ext::MPIProcessor *) peSrc;
    nanos::ext::MPIProcessor * dst = (nanos::ext::MPIProcessor *) peDst;
    int res;
    MPI_Comm_compare(src->getCommunicator(),dst->getCommunicator(),&res);
    //If both devices are in the same comunicator, they can do a dev2dev communication, otherwise go through host
    if (res == MPI_IDENT){
        //MPI_Status status;
        cacheOrder order;
        //Send the destination rank together with DEV2DEV OPID (we'll substract it on remote node)
        order.opId = OPID_DEVTODEV+dst->getRank();
        order.devAddr = (uint64_t) dst;
        order.hostAddr = (uint64_t) src;
        order.size = size;        
        //Send OPID_DEV2DEV (max OPID)+rank for one and -rank for the other
        //This way we can encode ranks inside OPID while keeping those OPID uniques
        //Send one to the source telling him what the send (+1) and to who
        nanos::ext::MPIProcessor::nanosMPISend(&order, 1, cacheStruct, src->getRank(), TAG_CACHE_ORDER, src->getCommunicator());
        order.opId = -src->getRank();
        //Send one to the dst telling him who's the source  (-1) and where to store
        nanos::ext::MPIProcessor::nanosMPISend(&order, 1, cacheStruct, dst->getRank(), TAG_CACHE_ORDER, dst->getCommunicator());
        //Wait for ACK from receiver
        //printf("espero ACK\n");
        short ans;
        nanos::ext::MPIProcessor::nanosMPIRecv(&ans, 1, MPI_SHORT, dst->getRank(), TAG_CACHE_ANSWER_DEV2DEV, dst->getCommunicator(), MPI_STATUS_IGNORE );
    } else {
        copyOut(cdDst,addrSrc,size,peSrc);
        copyIn(addrDst,cdDst,size,peDst);
    }
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return true;
}

void MPIDevice::initMPICacheStruct() {
    //Initialize cacheStruct in case it's not initialized
    if (cacheStruct == 0) {
        MPI_Datatype typelist[4] = {MPI_INT, MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG};
        int blocklen[4] = {1, 1, 1, 1};
        MPI_Aint disp[4] = {offsetof(cacheOrder, opId), offsetof(cacheOrder, hostAddr), offsetof(cacheOrder, devAddr), offsetof(cacheOrder, size)};
        MPI_Type_create_struct(4, blocklen, disp, typelist, &cacheStruct);
        MPI_Type_commit(&cacheStruct);
    }
}

void MPIDevice::taskPreInit(MPI_Comm& comm){
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_WAIT_FOR_COPIES_EVENT);
    executingTask=true;
    //int flag=0;
    //MPI_Status status;
    
    //if we can acquire, it means that there is no copy going on
    //while (doingCopyIn) { usleep(10000);};
    //Test if there are no copies pending (flag==0)
    //MPI_Iprobe( 0, TAG_CACHE_ORDER, comm, &flag, MPI_STATUS_IGNORE  );
//    while (flag!=0){
//          //Let the cache thread take the lock and keep working...
//          usleep(100);
//          //if we can acquire, it means that there is no copy going on
//          cacheLock.acquire();
//          //Test if there are no copies pending (flag==0)
//          MPI_Iprobe( 0, TAG_CACHE_ORDER, comm, &flag, MPI_STATUS_IGNORE  );
//          cacheLock.release();
//    }
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

void MPIDevice::taskPostFinish(MPI_Comm& comm){
    executingTask=false;
    //int flag=0;
    //MPI_Status status;
    
    //if we can acquire, it means that there is no copy going on
    //while (doingCopyIn) { usleep(10000);};
    //Test if there are no copies pending (flag==0)
    //MPI_Iprobe( 0, TAG_CACHE_ORDER, comm, &flag, MPI_STATUS_IGNORE  );
//    while (flag!=0){
//          //Let the cache thread take the lock and keep working...
//          usleep(100);
//          //if we can acquire, it means that there is no copy going on
//          cacheLock.acquire();
//          //Test if there are no copies pending (flag==0)
//          MPI_Iprobe( 0, TAG_CACHE_ORDER, comm, &flag, MPI_STATUS_IGNORE  );
//          cacheLock.release();
//    }
}

void MPIDevice::mpiCacheWorker() {
    //myThread = myThread->getNextThread();
    MPI_Comm parentcomm; /* intercommunicator */
    MPI_Comm_get_parent(&parentcomm);
    //If this process was not spawned, we don't need this daemon-thread
    if (parentcomm != 0 && parentcomm != MPI_COMM_NULL) {
        //MPI_Status status;
        //short ans=1;
        cacheOrder order;      
        MPI_Request recv_request;        
        int parentRank=nanos::ext::MPIProcessor::nanosMpiGetParentRank();
        MPI_Recv_init(&order, 1, cacheStruct, parentRank , TAG_CACHE_ORDER, parentcomm, &recv_request);
        while(1) {
            //if (!t->isRunning()) break; //{ //std::cerr << "FINISHING MPI THD!" << std::endl; break; }
            //if (sys.getNetwork()->getNodeNum() == 0  ) //std::cerr <<"ppp poll " << myThread->getId() << std::endl;
            //MPI_Comm_get_parent(&parentcomm);
            //TODO: Check if this is faster than sending ack's to the host  
            int pendingMessage=0;
            MPI_Iprobe(parentRank, TAG_CACHE_ORDER, parentcomm,&pendingMessage, MPI_STATUS_IGNORE);
            if (pendingMessage==0 && executingTask){
                myThread->yield();
            } else if (pendingMessage==1) {
                MPI_Start( &recv_request );
                MPI_Wait( &recv_request, MPI_STATUS_IGNORE );
        //printf("recv con tag %d, desde %d\n",tag,source);
                 // MPI_Recv(buf, count, datatype, source, tag, comm, MPI_STATUS_IGNORE );
    //            if (!(order.opId==OPID_COPYIN || order.opId>=OPID_DEVTODEV || order.opId<=0 )){
    //                doingCopyIn=false;
    //            }
                //TODO: make this operation single-message
                //and check performance with probe+remake struct datatype+single-message vs dual message
                switch (order.opId) {
                    case OPID_FINISH:
                        return;
                    case OPID_COPYIN:
                    {    
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_COPYIN_EVENT);
                        //std::cerr << "Hago un CopyIn en device\n";
                        //printf("Dir copyin%p, size %lu\n",(void*) order.devAddr,order.size);
                        //int incoming_msg_size;
                        //MPI_Probe(MPI_ANY_SOURCE, 98, parentcomm, MPI_STATUS_IGNORE );
                        //MPI_Get_count(MPI_STATUS_IGNORE , MPI_BYTE, &incoming_msg_size);
                        //MPI_Comm_get_parent(&parentcomm);
                        nanos::ext::MPIProcessor::nanosMPIRecv((void*) order.devAddr, order.size, MPI_BYTE, parentRank, TAG_CACHE_DATA_IN, parentcomm, MPI_STATUS_IGNORE );
                        DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
                        if (ent != NULL) 
                        { 
                           if (ent->getOwner() != NULL) {
                              ent->getOwner()->invalidate( *_masterDir, (uint64_t) order.devAddr, ent);
                           } else {
                              ent->increaseVersion();
                           }
                        }

                        //float* arr= (float*) order.devAddr;
                        //printf("Copio valor %f\n",arr[2047]);

                        //nanoxRegisterMPICopyIn(order.devAddr, order.hostAddr, order.size, order.opId);
                        //nanos::ext::MPIProcessor::nanos_MPI_Send(&ans, 1, MPI_SHORT, 0, TAG_CACHE_ANSWER_CIN, parentcomm);

                        //std::cerr << "Fin un CopyIn en device\n";
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }
                    case OPID_COPYOUT:
                    {
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_COPYOUT_EVENT);
                        //printf("Hago un copyOut de device %p\n",(void *) order.devAddr);
                        DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
                        if (ent != NULL) 
                        {
                           if (ent->getOwner() != NULL )
                              if ( !ent->isInvalidated() )
                              {
                                 std::list<uint64_t> tagsToInvalidate;
                                 tagsToInvalidate.push_back( ( uint64_t ) order.devAddr );
                                 _masterDir->synchronizeHost( tagsToInvalidate );
                              }
                        }
                        nanos::ext::MPIProcessor::nanosMPISend((void *) order.devAddr, order.size, MPI_BYTE, parentRank, TAG_CACHE_DATA_OUT, parentcomm);
                        //nanos::ext::MPIProcessor::nanos_MPI_Send(&ans, 1, MPI_SHORT, 0, TAG_CACHE_ANSWER_COUT, parentcomm);
                        //std::cerr << "Fin copyOut en device\n";
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }
                    case OPID_FREE:
                    {
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_FREE_EVENT);
                        //std::cerr << "Hago un free en device\n";
                        //nanoxRegisterMPIFree((void*)order.devAddr, order.size);
                        delete[] (char *) order.devAddr;
                        //printf("Dir free%p\n",(char*) order.devAddr);    
    //                    DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
    //                    if (ent != NULL) 
    //                    {
    //                        ent->setInvalidated(true);
    //                    }
                        //nanos::ext::MPIProcessor::nanos_MPI_Send(&ans, 1, MPI_SHORT, 0, TAG_CACHE_ANSWER_FREE, parentcomm);

                        //std::cerr << "Fin free en device\n";
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }
                    case OPID_ALLOCATE:           
                    {
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_ALLOC_EVENT);
                        //std::cerr << "Hago un allocate en device\n";
                        char* ptr = new (std::nothrow) char[order.size];
                        order.devAddr = (uint64_t) ptr;
                        //printf("Dir alloc%p size %lu\n",(void*) order.devAddr,order.size);
                        //MPI_Comm_get_parent(&parentcomm);
                        nanos::ext::MPIProcessor::nanosMPISend(&order, 1, cacheStruct, parentRank, TAG_CACHE_ANSWER_ALLOC, parentcomm);
                        //std::cerr << "Fin allocate en device\n";
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }    
                    case OPID_REALLOC:           
                    {
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_REALLOC_EVENT);
                        //std::cerr << "Hago un reallocate en device\n";
                        delete[] (char *) order.devAddr;
                        char* ptr = new char[order.size];
                        ////printf("Realloc %d, %d\n", order.size,order.old_size);
                        ////printf("Copio de %p a %p, tam %d\n",(void*)  order.devAddr, ptr, order.old_size);
                        //printf("Copio valor %p\n",((void*)  order.devAddr));                    
                        //TODO: CHECK THIS, it's probably wrong
                        DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) ptr );
                        if (ent != NULL) 
                        { 
                           if (ent->getOwner() != NULL) 
                           {
                              ent->getOwner()->deleteEntry((uint64_t) ptr, order.size);
                           }
                        }
                        //memcpy(ptr, (void*)  order.devAddr, order.old_size);
                        order.devAddr = (uint64_t) ptr;
                        //MPI_Comm_get_parent(&parentcomm);
                        nanos::ext::MPIProcessor::nanosMPISend(&order, 1, cacheStruct, parentRank, TAG_CACHE_ANSWER_REALLOC, parentcomm);
                        //std::cerr << "Fin reallocate en device\n";
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }
                    case OPID_COPYLOCAL:
                    {
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_COPYLOCAL_EVENT);
                        //std::cerr << "Hago un copylocal en device\n";
                        memcpy((void*) order.devAddr, (void*) order.hostAddr, order.size);
                        //nanos::ext::MPIProcessor::nanos_MPI_Send(&ans, 1, MPI_SHORT, 0, TAG_CACHE_ANSWER, parentcomm);
                        //std::cerr << "Fin un copylocal en device\n";
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }
                    //If not a fixed code, its a dev2dev copy where i act as the source
                    default:
                    {
                        //printf("entro en default %d\n", order.opId);
                        //Opid >= DEV2DEV (largest OPID) is dev2dev+rank
                        if (order.opId>=OPID_DEVTODEV){
                            NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_DEV2DEV_OUT_EVENT);
                            //std::cerr << "Hago un dev 2  sender device " << order.opId-OPID_DEVTODEV << "\n";
                            //Get the rank
                            int dstRank=order.opId-OPID_DEVTODEV;
                            //MPI_Comm_get_parent(&parentcomm);
                            nanos::ext::MPIProcessor::nanosMPISend((void *) order.hostAddr, order.size, MPI_BYTE, dstRank, TAG_CACHE_DEV2DEV, MPI_COMM_WORLD);
                            //std::cerr << "Fin dev2dev spource en device\n";
                            NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        //Opid <= 0 (largest OPID) is -rank (in a dev2dev communication)
                        } else if (order.opId<=0) {
                            NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_DEV2DEV_IN_EVENT);
                            DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
                            if (ent != NULL) 
                            { 
                               if (ent->getOwner() != NULL) {
                                  ent->getOwner()->invalidate( *_masterDir, (uint64_t) order.devAddr, ent);
                               } else {
                                  ent->increaseVersion();
                               }
                            }
                            //std::cerr << "Hago un dev 2 recv device " << -order.opId << "\n";
                            //Do the inverse operation of the host
                            int srcRank=-order.opId;

                            nanos::ext::MPIProcessor::nanosMPIRecv((void *) order.devAddr, order.size, MPI_BYTE, srcRank, TAG_CACHE_DEV2DEV, MPI_COMM_WORLD, MPI_STATUS_IGNORE );                        
                            //Send ACK to host
                            short ans;
                            nanos::ext::MPIProcessor::nanosMPISend(&ans, 1, MPI_SHORT, parentRank, TAG_CACHE_ANSWER_DEV2DEV, parentcomm);
                            //std::cerr << "Fin dev2dev receiver en device\n";
                            NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        }
                        break;
                  }
               }
            //if ( sys.getNetwork()->getNodeNum() == 0 ) //std::cerr <<"ppp poll complete " << myThread->getId() << std::endl;
            //if ( sys.getNetwork()->getNodeNum() == 0 ) //std::cerr << "thread change! " << std::endl;
            //myThread = myThread->getNextThread();
           }
        }
    }
}