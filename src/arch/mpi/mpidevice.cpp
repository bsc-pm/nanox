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
#include "mpiprocessor_decl.hpp"
#include "workdescriptor_decl.hpp"
#include "processingelement_fwd.hpp"
#include "copydescriptor_decl.hpp"
#include "mpidevice.hpp"
#include <unistd.h>
#include "deviceops.hpp"


//TODO: check for errors in communications
//TODO: Depending on multithread level, wait for answers or not
using namespace nanos;


MPI_Datatype MPIDevice::cacheStruct;
char MPIDevice::_executingTask=0;
bool MPIDevice::_createdExtraWorkerThread=false;

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
void * MPIDevice::memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, WorkDescriptor const &wd, unsigned int copyIdx ) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_ALLOC_EVENT);
    //std::cerr << "Inicio allocate\n";
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) &mem.getConstPE();
    cacheOrder order;
    order.opId = OPID_ALLOCATE;
    order.hostAddr = 0;
    order.size = size;
    //MPI_Status status;
    nanos::ext::MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_M2S_ORDER, myPE->getCommunicator());
    nanos::ext::MPIRemoteNode::nanosMPIRecv(&order, 1, cacheStruct, myPE->getRank(), TAG_CACHE_ANSWER_ALLOC, myPE->getCommunicator(), MPI_STATUS_IGNORE );
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    //std::cerr << "Fin allocate\n";
    if (order.devAddr==0){
        return NULL;
    }
    return (void *) order.devAddr;
}

/* \brief free address
 */
//void memFree( uint64_t addr, SeparateMemoryAddressSpace &mem ) const;
void MPIDevice::memFree( uint64_t addr, SeparateMemoryAddressSpace &mem ) {
    if (addr == 0) return;
    //std::cerr << "Inicio free\n";
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_FREE_EVENT);
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) &mem.getConstPE();
    cacheOrder order;
    order.opId = OPID_FREE;
    order.devAddr = addr;
    order.size = 0;
    //short ans;
    //MPI_Status status;    
    nanos::ext::MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_M2S_ORDER, myPE->getCommunicator());
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    //nanos::ext::MPIProcessor::nanos_MPI_Recv(&ans, 1, MPI_SHORT, myPE->getRank(), TAG_CACHE_ANSWER_FREE, myPE->_communicator, MPI_STATUS_IGNORE );

    //std::cerr << "Fin free\n";
}

size_t MPIDevice::getMemCapacity( SeparateMemoryAddressSpace const &mem ) const{
    //MAXSIZE-1
    return (size_t)-1;
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
    nanos::ext::MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_M2S_ORDER, myPE->getCommunicator());
    nanos::ext::MPIRemoteNode::nanosMPIRecv(&order, 1, cacheStruct, myPE->getRank(), TAG_CACHE_ANSWER_REALLOC, myPE->getCommunicator(), MPI_STATUS_IGNORE );

    //std::cerr << "Fin realloc\n";
    //printf("Copiado de %p\n",(void *) order.devAddr);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return (void *) order.devAddr;
}

/* \brief Copy from remoteSrc in the host to localDst in the device
 *        Returns true if the operation is synchronous
 */
void MPIDevice::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_COPYIN_SYNC_EVENT);
    //std::cerr << "Inicio copyin\n";
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) &mem.getConstPE();
    cacheOrder order;
    order.opId = OPID_COPYIN;
    order.devAddr = devAddr;
    order.hostAddr = hostAddr;
    order.size = len;
    //short ans;
    //MPI_Status status;
    MPI_Request req;
    //printf("Dir copyin host%p a device %p, size %lu\n",(void*) order.hostAddr,(void*) order.devAddr,order.size);
    nanos::ext::MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_M2S_ORDER, myPE->getCommunicator());
    nanos::ext::MPIRemoteNode::nanosMPIIsend((void*) order.hostAddr, order.size, MPI_BYTE, myPE->getRank(), TAG_CACHE_DATA_IN, myPE->getCommunicator(), &req);
    //Free the request (we no longer care about when it finishes, offload process will take care of that)
    //MPI_Request_free(&req);
    myPE->appendToPendingRequests(req);
    //nanos::ext::MPIProcessor::nanos_MPI_Recv(&ans, 1, MPI_SHORT, myPE->getRank(), TAG_CACHE_ANSWER_CIN, myPE->_communicator, MPI_STATUS_IGNORE );
    //std::cerr << "Fin copyin\n";
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

/* \brief Copy from localSrc in the device to remoteDst in the host
 *        Returns true if the operation is synchronous
 */
void MPIDevice::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_COPYOUT_SYNC_EVENT);
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) &mem.getConstPE();
    cacheOrder order;
    //if PE is executing something, this means an extra cache-thread could be usefull, send creation signal
    if (myPE->getCurrExecutingWd()!=NULL && !myPE->getHasWorkerThread()) {        
      order.opId = OPID_CREATEAUXTHREAD;
      nanos::ext::MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_M2S_ORDER, myPE->getCommunicator());
      myPE->setHasWorkerThread(true);
    }
    order.opId = OPID_COPYOUT;
    order.devAddr = (uint64_t) devAddr;
    order.hostAddr = (uint64_t) hostAddr;    
    order.size = len;
    //MPI_Status status;
    //TODO: Make this async
    nanos::ext::MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_M2S_ORDER, myPE->getCommunicator());
    nanos::ext::MPIRemoteNode::nanosMPIRecv((void*) order.hostAddr, order.size, MPI_BYTE, myPE->getRank(), TAG_CACHE_DATA_OUT, myPE->getCommunicator(), MPI_STATUS_IGNORE );
    //short ans;
    //nanos::ext::MPIProcessor::nanos_MPI_Recv(&ans, 1, MPI_SHORT, myPE->getRank(), TAG_CACHE_ANSWER_COUT, myPE->_communicator, MPI_STATUS_IGNORE );
    //std::cerr << "Fin copyout\n";
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

///* \brief Copy localy in the device from src to dst
// */
//void MPIDevice::copyLocal(void *dst, void *src, size_t size, ProcessingElement *pe) {
//    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_COPYLOCAL_SYNC_EVENT);
//    //std::cerr << "Inicio copylocal\n";
//    //HostAddr will be src and DevAddr will be dst (both are device addresses)
//    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
//    cacheOrder order;
//    order.opId = OPID_COPYLOCAL;
//    order.devAddr = (uint64_t) dst;
//    order.hostAddr = (uint64_t) src;
//    order.size = size;
//    nanos::ext::MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_CACHE_ORDER, myPE->getCommunicator());
//    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
//    //std::cerr << "Fin copyin\n";
//}


bool MPIDevice::_copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_COPYDEV2DEV_SYNC_EVENT);
    //This will never be another PE type which is not MPI (or something is broken in core :) )
    nanos::ext::MPIProcessor * src = (nanos::ext::MPIProcessor *) &memOrig.getConstPE();
    nanos::ext::MPIProcessor * dst = (nanos::ext::MPIProcessor *) &memDest.getConstPE();
    int res;
    MPI_Comm_compare(src->getCommunicator(),dst->getCommunicator(),&res);
    //If both devices are in the same comunicator, they can do a dev2dev communication, otherwise go through host
    if (res == MPI_IDENT){
        ops->addOp();
        cacheOrder order;
        //if PE is executing something, this means an extra cache-thread could be usefull, send creation signal
        if (src->getCurrExecutingWd()!=NULL && !src->getHasWorkerThread()) {        
          order.opId = OPID_CREATEAUXTHREAD;
          nanos::ext::MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, src->getRank(), TAG_M2S_ORDER, src->getCommunicator());
          src->setHasWorkerThread(true);
        }
        //Send the destination rank together with DEV2DEV OPID (we'll substract it on remote node)
        order.opId = OPID_DEVTODEV+dst->getRank();
        order.devAddr = (uint64_t) devDestAddr;
        order.hostAddr = (uint64_t) devOrigAddr;
        order.size = len;        
        //Send OPID_DEV2DEV (max OPID)+rank for one and -rank for the other
        //This way we can encode ranks inside OPID while keeping those OPID uniques
        //Send one to the source telling him what the send (+1) and to who
        nanos::ext::MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, src->getRank(), TAG_M2S_ORDER, src->getCommunicator());
        order.opId = -src->getRank();
        //Send one to the dst telling him who's the source  (-1) and where to store
        nanos::ext::MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, dst->getRank(), TAG_M2S_ORDER, dst->getCommunicator());
        ops->completeOp(); 
        if ( f ) {
           (*f)(); 
        }
        return true;
        //Wait for ACK from receiver
        //printf("espero ACK\n");
        //short ans;
        //nanos::ext::MPIRemoteNode::nanosMPIRecv(&ans, 1, MPI_SHORT, dst->getRank(), TAG_CACHE_ANSWER_DEV2DEV, dst->getCommunicator(), MPI_STATUS_IGNORE );
    }
    return false;
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
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

static void createExtraCacheThread(){    
    //Create extra worker thread
    MPI_Comm mworld= MPI_COMM_WORLD;
    ext::SMPProcessor *core = sys.getSMPPlugin()->getLastFreeSMPProcessorAndReserve();
    if (core==NULL) {
        core = sys.getSMPPlugin()->getSMPProcessorByNUMAnode(0,nanos::ext::MPIRemoteNode::getCurrentProcessor());
    }
    MPIProcessor *mpi = NEW nanos::ext::MPIProcessor(&mworld, CACHETHREADRANK,-1, false, false, /* Dummy*/ MPI_COMM_SELF, core, /* Dummmy memspace */ 0);
    nanos::ext::MPIDD * dd = NEW nanos::ext::MPIDD((nanos::ext::MPIDD::work_fct) MPIDevice::remoteNodeCacheWorker);
    WD* wd = NEW WD(dd);
    NANOS_INSTRUMENT( sys.getInstrumentation()->incrementMaxThreads(); )
    mpi->startMPIThread(wd);
}

void MPIDevice::remoteNodeCacheWorker() {                            
    //myThread = myThread->getNextThread();
    MPI_Comm parentcomm; /* intercommunicator */
    MPI_Comm_get_parent(&parentcomm);    
    const size_t alignThreshold = nanos::ext::MPIProcessor::getAlignThreshold();
    const size_t alignment = nanos::ext::MPIProcessor::getAlignment();
    //If this process was not spawned, we don't need this daemon-thread
    if (parentcomm != 0 && parentcomm != MPI_COMM_NULL) {
        MPI_Status status;
        //short ans=1;
        cacheOrder order;  
        int parentRank=0;
        int firstParent=-1;
        while(1) {
                if (_createdExtraWorkerThread) {
                    int flag=0;
                    //When this is not the executer thread, perform async probes
                    //as performing sync probes will lock MPI implementation (at least @ IMPI)
                    while (flag==0 && !nanos::ext::MPIRemoteNode::getDisconnectedFromParent()) {
                        MPI_Iprobe(MPI_ANY_SOURCE,TAG_M2S_ORDER,parentcomm,&flag,MPI_STATUS_IGNORE);
                        if (flag==0) usleep(50000);
                    }
                }
                
                if (!nanos::ext::MPIRemoteNode::getDisconnectedFromParent())
                    MPI_Recv(&order, 1, cacheStruct, MPI_ANY_SOURCE, TAG_M2S_ORDER, parentcomm, &status);
                else
                    order.opId=OPID_FINISH;
 
                parentRank=status.MPI_SOURCE;
                switch (order.opId) {
                    case OPID_TASK_INIT:
                    {
                        int task_id=(int)order.hostAddr; 
                        //If our node is used by more than one parent
                        //create the worker thread as we may have cache orders from
                        //one node while executing tasks from the other node
                        if (firstParent!=parentRank && !_createdExtraWorkerThread) {
                            if (firstParent==-1) {
                                firstParent=parentRank;
                            } else {
                                _createdExtraWorkerThread=true;  
                                createExtraCacheThread();
                                //Create extra thread which controls the cache, add current task to queue and become executor
                                nanos::ext::MPIRemoteNode::addTaskToQueue(task_id,parentRank);
                                nanos::ext::MPIRemoteNode::nanosMPIWorker();
                                return;
                            }
                        }
                        if (_createdExtraWorkerThread) {
                            //Add task to execution queue
                            nanos::ext::MPIRemoteNode::addTaskToQueue(task_id,parentRank);
                        } else {                       
                            //Execute the task in current thread
                            nanos::ext::MPIRemoteNode::setCurrentTaskParent(parentRank);
                            nanos::ext::MPIRemoteNode::executeTask(task_id);
                        }
                        break;
                    }
                    case OPID_CREATEAUXTHREAD: 
                    {
                        //Only create once (multiple requests may come
                        //since different parents do not query the status of the thread)
                        //ignore extra ones
                        if (!_createdExtraWorkerThread) {
                            _createdExtraWorkerThread=true;  
                            createExtraCacheThread();
                            nanos::ext::MPIRemoteNode::nanosMPIWorker();
                            return;
                        }
                        break;
                    }
                    case OPID_FINISH:    
                    {
                        if (_createdExtraWorkerThread) {
                           //Add finish task to execution queue
                           nanos::ext::MPIRemoteNode::addTaskToQueue(TASK_END_PROCESS,parentRank);   
                           nanos::ext::MPIRemoteNode::getTaskLock().acquire(); 
                           nanos::ext::MPIRemoteNode::getTaskLock().acquire(); 
                        } else {							
                           //Execute in current thread
                           nanos::ext::MPIRemoteNode::setCurrentTaskParent(parentRank);
                           nanos::ext::MPIRemoteNode::executeTask(TASK_END_PROCESS);   
                        }
                        return;
                    }
                    case OPID_UNIFIED_MEM_REQ:
                    {
                        nanos::ext::MPIRemoteNode::unifiedMemoryMallocRemote(order, parentRank, parentcomm);
                        break;
                    }
                    case OPID_COPYIN:
                    {    
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_COPYIN_EVENT);
                        nanos::ext::MPIRemoteNode::nanosMPIRecv((void*) order.devAddr, order.size, MPI_BYTE, parentRank, TAG_CACHE_DATA_IN, parentcomm, MPI_STATUS_IGNORE );
                        //                            DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
//                            if (ent != NULL) 
//                            { 
//                               if (ent->getOwner() != NULL) {
//                                  ent->getOwner()->invalidate( *_masterDir, (uint64_t) order.devAddr, ent);
//                               } else {
//                                  ent->increaseVersion();
//                               }
//                            }
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }
                    case OPID_COPYOUT:
                    {
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_COPYOUT_EVENT);
//                            DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
//                            if (ent != NULL) 
//                            {
//                               if (ent->getOwner() != NULL )
//                                  if ( !ent->isInvalidated() )
//                                  {
//                                     std::list<uint64_t> tagsToInvalidate;
//                                     tagsToInvalidate.push_back( ( uint64_t ) order.devAddr );
//                                     _masterDir->synchronizeHost( tagsToInvalidate );
//                                  }
//                            }
                        nanos::ext::MPIRemoteNode::nanosMPISend((void *) order.devAddr, order.size, MPI_BYTE, parentRank, TAG_CACHE_DATA_OUT, parentcomm);
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }
                    case OPID_FREE:
                    {
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_FREE_EVENT);
                        std::free((char *) order.devAddr);
                        //printf("Dir free%p\n",(char*) order.devAddr);    
    //                    DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
    //                    if (ent != NULL) 
    //                    {
    //                        ent->setInvalidated(true);
    //                    }
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }
                    case OPID_ALLOCATE:           
                    {
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_ALLOC_EVENT);
                        char* ptr;                        
                        if (order.size<alignThreshold){
                           ptr = (char*) malloc(order.size);
                        } else {
                           posix_memalign((void**)&ptr,alignment,order.size);
                        }
                        order.devAddr = (uint64_t) ptr;    
                        nanos::ext::MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, parentRank, TAG_CACHE_ANSWER_ALLOC, parentcomm);
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }    
                    case OPID_REALLOC:           
                    {
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_REALLOC_EVENT);
                        std::free((char *) order.devAddr);
                        char* ptr;                        
                        if (order.size<alignThreshold){
                           ptr = (char*) malloc(order.size);
                        } else {
                           posix_memalign((void**)&ptr,alignment,order.size);
                        }
//                            DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) ptr );
//                            if (ent != NULL) 
//                            { 
//                               if (ent->getOwner() != NULL) 
//                               {
//                                  ent->getOwner()->deleteEntry((uint64_t) ptr, order.size);
//                               }
//                            }
                        order.devAddr = (uint64_t) ptr;
                        nanos::ext::MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, parentRank, TAG_CACHE_ANSWER_REALLOC, parentcomm);
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }
                    case OPID_COPYLOCAL:
                    {
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_COPYLOCAL_EVENT);
                        memcpy((void*) order.devAddr, (void*) order.hostAddr, order.size);
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }
                    //If not a fixed code, its a dev2dev copy
                    default:
                    {
                        //Opid >= DEV2DEV (largest OPID) is dev2dev+rank and im source
                        if (order.opId>=OPID_DEVTODEV){
                            NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_DEV2DEV_OUT_EVENT);
                            //Get the rank
                            int dstRank=order.opId-OPID_DEVTODEV;
                            //MPI_Comm_get_parent(&parentcomm);
                            nanos::ext::MPIRemoteNode::nanosMPISend((void *) order.hostAddr, order.size, MPI_BYTE, dstRank, TAG_CACHE_DEV2DEV, MPI_COMM_WORLD);
                            NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        //Opid <= 0 (smallest OPID) is -rank (in a dev2dev communication) and im destination
                        } else if (order.opId<=0) {
                            NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_DEV2DEV_IN_EVENT);
//                                DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
//                                if (ent != NULL) 
//                                { 
//                                   if (ent->getOwner() != NULL) {
//                                      ent->getOwner()->invalidate( *_masterDir, (uint64_t) order.devAddr, ent);
//                                   } else {
//                                      ent->increaseVersion();
//                                   }
//                                }
                            //Do the inverse operation of the host
                            int srcRank=-order.opId;

                            nanos::ext::MPIRemoteNode::nanosMPIRecv((void *) order.devAddr, order.size, MPI_BYTE, srcRank, TAG_CACHE_DEV2DEV, MPI_COMM_WORLD, MPI_STATUS_IGNORE );                        
                            NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        }
                        break;
                  }
           }
        }
    }
}
