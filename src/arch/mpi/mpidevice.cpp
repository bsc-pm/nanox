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
#include "request.hpp"
#include "cacheorder.hpp"

//TODO: check for errors in communications
//TODO: Depending on multithread level, wait for answers or not
using namespace nanos;
using namespace nanos::ext;

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

    mpi::command::Allocate order( reinterpret_cast<MPIProcessor const&>(mem.getConstPE()), size );
    order.execute();

    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    //std::cerr << "Fin allocate\n";

    return order.getDeviceAddress(); // May be null
}

/* \brief free address
 */
//void memFree( uint64_t addr, SeparateMemoryAddressSpace &mem ) const;
void MPIDevice::memFree( uint64_t addr, SeparateMemoryAddressSpace &mem ) {
    if (addr == 0) return;

    //std::cerr << "Inicio free\n";
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_FREE_EVENT);

    mpi::command::Free order( reinterpret_cast<MPIProcessor const&>(mem.getConstPE()), address );
    order.execute();

    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    //std::cerr << "Fin free\n";
}

size_t MPIDevice::getMemCapacity( SeparateMemoryAddressSpace &mem ) {
    //MAXSIZE-1
    return (size_t)-1;
}

/* \brief Reallocate and copy from address.
 */
void * MPIDevice::realloc(void *address, size_t size, size_t old_size, ProcessingElement *pe) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_REALLOC_EVENT);
    //std::cerr << "Inicio REALLOC\n";

    mpi::command::Realloc order( reinterpret_cast<MPIProcessor const&>(mem.getConstPE()), address, size );
    order.execute();

    //std::cerr << "Fin realloc\n";
    //printf("Copiado de %p\n",(void *) order.devAddr);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;

    return order.getDeviceAddress();
}

/* \brief Copy from remoteSrc in the host to localDst in the device
 *        Returns true if the operation is synchronous
 */
void MPIDevice::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_COPYIN_SYNC_EVENT);
    //std::cerr << "Inicio copyin\n";

    mpi::command::CopyIn order( reinterpret_cast<MPIProcessor const&>(mem.getConstPE()), hostAddr, devAddr, len );
    order.execute();

    //std::cerr << "Fin copyin\n";
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

/* \brief Copy from localSrc in the device to remoteDst in the host
 *        Returns true if the operation is synchronous
 */
void MPIDevice::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_COPYOUT_SYNC_EVENT);

    MPIProcessor &destination = reinterpret_cast<MPIProcessor &>( mem.getPE() );
    //if PE is executing something, this means an extra cache-thread could be usefull, send creation signal
    if ( destination.getCurrExecutingWd() != NULL && !destination.getHasWorkerThread()) {        
    	  mpi::command::CreateAuxiliaryThread createThread( destination );
		  createThread.execute();

        destination.setHasWorkerThread(true);
    }

    mpi::command::CopyOut copyOrder( reinterpret_cast<MPIProcessor const&>(mem.getConstPE()), hostAddr, devAddr, len );
    copyOrder.execute();

    //std::cerr << "Fin copyout\n";
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

///* \brief Copy localy in the device from src to dst
// */
//void MPIDevice::copyLocal(void *dst, void *src, size_t size, ProcessingElement *pe) {
//    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_COPYLOCAL_SYNC_EVENT);
//    //std::cerr << "Inicio copylocal\n";
//    //HostAddr will be src and DevAddr will be dst (both are device addresses)
//    MPIProcessor * myPE = (MPIProcessor *) pe;
//    cacheOrder order;
//    order.opId = OPID_COPYLOCAL;
//    order.devAddr = (uint64_t) dst;
//    order.hostAddr = (uint64_t) src;
//    order.size = size;
//    MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, myPE->getRank(), TAG_CACHE_ORDER, myPE->getCommunicator());
//    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
//    //std::cerr << "Fin copyin\n";
//}


bool MPIDevice::_copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_COPYDEV2DEV_SYNC_EVENT);
    //This will never be another PE type which is not MPI (or something is broken in core :) )
    MPIProcessor &src = reinterpret_cast<MPIProcessor &>( memOrig.getPE() );
    MPIProcessor &dst = reinterpret_cast<MPIProcessor &>( memDest.getPE() );
    int res;
    MPI_Comm_compare(src.getCommunicator(), dst.getCommunicator(), &res);
    //If both devices are in the same comunicator, they can do a dev2dev communication, otherwise go through host
    if (res == MPI_IDENT){
        ops->addOp();

        //if PE is executing something, this means an extra cache-thread could be usefull, send creation signal
        if ( destination.getCurrExecutingWd() != NULL && !destination.getHasWorkerThread()) {        
        	   mpi::command::CreateAuxiliaryThread createThread( destination );
	         createThread.execute();

            destination.setHasWorkerThread(true);
        }

        //Send the destination rank together with DEV2DEV OPID (we'll substract it on remote node)
        mpi::command::CopyDeviceToDevice order( src, dst, devOrigAddr, devDstAddr, len );
        order.execute();

        ops->completeOp(); 
        if ( f ) {
           (*f)(); 
        }
        return true;
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
        core = sys.getSMPPlugin()->getSMPProcessorByNUMAnode(0,MPIRemoteNode::getCurrentProcessor());
    }
    MPIProcessor *mpi = NEW MPIProcessor(&mworld, CACHETHREADRANK,-1, false, false, /* Dummy*/ MPI_COMM_SELF, core, /* Dummmy memspace */ 0);
    MPIDD * dd = NEW MPIDD((MPIDD::work_fct) MPIDevice::remoteNodeCacheWorker);
    WD* wd = NEW WD(dd);
    NANOS_INSTRUMENT( sys.getInstrumentation()->incrementMaxThreads(); )
    mpi->startMPIThread(wd);
}

void MPIDevice::remoteNodeCacheWorker() {                            
    //myThread = myThread->getNextThread();
    MPI_Comm parentcomm; /* intercommunicator */
    MPI_Comm_get_parent(&parentcomm);    
    const size_t alignThreshold = MPIProcessor::getAlignThreshold();
    const size_t alignment = MPIProcessor::getAlignment();
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
                    while (flag==0 && !MPIRemoteNode::getDisconnectedFromParent()) {
                        MPI_Iprobe(MPI_ANY_SOURCE,TAG_M2S_ORDER,parentcomm,&flag,MPI_STATUS_IGNORE);
                        if (flag==0) usleep(50000);
                    }
                }
                
                if (!MPIRemoteNode::getDisconnectedFromParent())
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
                                MPIRemoteNode::addTaskToQueue(task_id,parentRank);
                                MPIRemoteNode::nanosMPIWorker();
                                return;
                            }
                        }
                        if (_createdExtraWorkerThread) {
                            //Add task to execution queue
                            MPIRemoteNode::addTaskToQueue(task_id,parentRank);
                        } else {                       
                            //Execute the task in current thread
                            MPIRemoteNode::setCurrentTaskParent(parentRank);
                            MPIRemoteNode::executeTask(task_id);
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
                            MPIRemoteNode::nanosMPIWorker();
                            return;
                        }
                        break;
                    }
                    case OPID_FINISH:    
                    {
                        if (_createdExtraWorkerThread) {
                           //Add finish task to execution queue
                           MPIRemoteNode::addTaskToQueue(TASK_END_PROCESS,parentRank);   
                           MPIRemoteNode::getTaskLock().acquire(); 
                           MPIRemoteNode::getTaskLock().acquire(); 
                        } else {							
                           //Execute in current thread
                           MPIRemoteNode::setCurrentTaskParent(parentRank);
                           MPIRemoteNode::executeTask(TASK_END_PROCESS);   
                        }
                        return;
                    }
                    case OPID_UNIFIED_MEM_REQ:
                    {
                        MPIRemoteNode::unifiedMemoryMallocRemote(order, parentRank, parentcomm);
                        break;
                    }
                    case OPID_COPYIN:
                    {    
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_COPYIN_EVENT);
                        MPIRemoteNode::nanosMPIRecv((void*) order.devAddr, order.size, MPI_BYTE, parentRank, TAG_CACHE_DATA_IN, parentcomm, MPI_STATUS_IGNORE );
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
                        MPIRemoteNode::nanosMPISend((void *) order.devAddr, order.size, MPI_BYTE, parentRank, TAG_CACHE_DATA_OUT, parentcomm);
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
						ptr = NULL;
                        if (order.size<alignThreshold){
                           ptr = (char*) malloc(order.size);
                        } else {
                           posix_memalign((void**)&ptr,alignment,order.size);
                        }
                        order.devAddr = (uint64_t) ptr;    
                        MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, parentRank, TAG_CACHE_ANSWER_ALLOC, parentcomm);
                        NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        break;
                    }    
                    case OPID_REALLOC:           
                    {
                        NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_REALLOC_EVENT);
                        std::free((char *) order.devAddr);
                        char* ptr;
						ptr = NULL;                        
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
                        MPIRemoteNode::nanosMPISend(&order, 1, cacheStruct, parentRank, TAG_CACHE_ANSWER_REALLOC, parentcomm);
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
                            MPIRemoteNode::nanosMPISend((void *) order.hostAddr, order.size, MPI_BYTE, dstRank, TAG_CACHE_DEV2DEV, MPI_COMM_WORLD);
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

                            MPIRemoteNode::nanosMPIRecv((void *) order.devAddr, order.size, MPI_BYTE, srcRank, TAG_CACHE_DEV2DEV, MPI_COMM_WORLD, MPI_STATUS_IGNORE );                        
                            NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
                        }
                        break;
                  }
           }
        }
    }
}
