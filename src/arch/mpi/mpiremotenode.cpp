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

#include "mpiremotenode.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "config.hpp"
#include "mpithread.hpp"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include "mpi.h"

using namespace nanos;
using namespace nanos::ext;

extern __attribute__((weak)) int ompss_mpi_masks[];
extern __attribute__((weak)) unsigned int ompss_mpi_filenames[];
extern __attribute__((weak)) unsigned int ompss_mpi_file_sizes[];
extern __attribute__((weak)) unsigned int ompss_mpi_file_ntasks[];
extern __attribute__((weak)) void *ompss_mpi_func_pointers_host[];
extern __attribute__((weak)) void *ompss_mpi_func_pointers_dev[];



bool MPIRemoteNode::executeTask(int taskId) {
    bool ret=false;
    if (taskId==TASK_END_PROCESS){
       nanosMPIFinalize();
       nanos::ext::MPIRemoteNode::getTaskLock().release();
       ret=true;
    } else {
       void (* function_pointer)()=(void (*)()) ompss_mpi_func_pointers_dev[taskId];
       //nanos::MPIDevice::taskPreInit();
       function_pointer();
       //nanos::MPIDevice::taskPostFinish();
    }
    return ret;
}

int MPIRemoteNode::nanosMPIWorker(){
	bool finalize=false;
	while(!finalize){
		//Acquire twice and block until cache thread unlocks
      testTaskQueueSizeAndLock();
      setCurrentTaskParent(getQueueCurrentTaskParent());
		finalize=executeTask(getQueueCurrTaskIdentifier());
      removeTaskFromQueue();
	}
   return 0;
}

void MPIRemoteNode::preInit(){
    nanosMPIInit(0,0,MPI_THREAD_MULTIPLE,0);
    nanos::ext::MPIRemoteNode::nanosSyncDevPointers(ompss_mpi_masks, ompss_mpi_filenames, ompss_mpi_file_sizes,ompss_mpi_file_ntasks,ompss_mpi_func_pointers_dev);
}

void MPIRemoteNode::mpiOffloadSlaveMain(){
    nanos::MPIDevice::remoteNodeCacheWorker();
    exit(0);
}

int MPIRemoteNode::ompssMpiGetFunctionIndexHost(void* func_pointer){
    int i;
    //This function WILL find the pointer, if it doesnt, program would crash anyways so I won't limit it
    for (i=0;ompss_mpi_func_pointers_host[i]!=func_pointer;i++) { }
    return i;
}


int MPIRemoteNode::ompssMpiGetFunctionIndexDevice(void* func_pointer){
    int i;
    //This function WILL find the pointer, if it doesnt, program would crash anyways so I won't limit it
    for (i=0;ompss_mpi_func_pointers_dev[i]!=func_pointer;i++) { }
    return i;
}

/**
 * Statics (mostly external API adapters provided to user or used by mercurium) begin here
 */


/**
 * All this tasks redefine nanox messages
 */
void MPIRemoteNode::nanosMPIInit(int *argc, char ***argv, int userRequired, int* userProvided) {
    if (_initialized) return;
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_INIT_EVENT);
    verbose0( "loading MPI support" );
    //Unless user specifies otherwise, enable blocking mode in MPI
    if (getenv("I_MPI_WAIT_MODE")==NULL) putenv(const_cast<char*> ("I_MPI_WAIT_MODE=1"));

    //If we are not offload slaves, initialice MPI plugin
    bool imSlave=getenv("OMPSS_OFFLOAD_SLAVE")!=NULL;
    if (!imSlave){
        if ( !sys.loadPlugin( "arch-mpi" ) )
          fatal0 ( "Couldn't load MPI support" );
    }

    _initialized=true;
    int provided;
    //If user provided a null pointer, we'll a value for internal checks
    if (userProvided==NULL) userProvided=&provided;

    int initialized;
    MPI_Initialized(&initialized);
    //In case it was already initialized (shouldn't happen, since we theorically "rename" the calls with mercurium), we won't try to do so
    //We'll trust user criteria, but show a warning
    if (!initialized) {
        if (userRequired != MPI_THREAD_MULTIPLE) {
            warning0("Initializing MPI with MPI_THREAD_MULTIPLE instead of user required mode, this is a requeriment for OmpSs offload");
        }
        MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, userProvided);
    } else {
        //Do not initialise, but check thread level and return the right provided value to the user
        MPI_Query_thread(userProvided);
    }
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myRank);

    //Remove possible trashfiles from other executions
    if (myRank==0 && !imSlave) {
       std::string lockname="./.ompssOffloadLock";
       remove(const_cast<char*> (lockname.c_str()));
       if (!nanos::ext::MPIProcessor::getMpiControlFile().empty()) remove(const_cast<char*> (nanos::ext::MPIProcessor::getMpiControlFile().c_str()));
    }


    nanos::MPIDevice::initMPICacheStruct();

    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

void MPIRemoteNode::nanosMPIFinalize() {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_FINALIZE_EVENT);
    for ( std::vector<MPI_Datatype*>::iterator it = _taskStructsCache.begin();
            it!=_taskStructsCache.end(); ++it ) {
        delete *it;
    }
    int resul;
    MPI_Finalized(&resul);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    if (!resul){
      //Free every node before finalizing
      DEEP_Booster_free(NULL,-1);
      MPI_Finalize();
    }
}

//TODO: Finish implementing shared memory
#define N_FREE_SLOTS 10
void MPIRemoteNode::unifiedMemoryMallocHost(size_t size, MPI_Comm communicator) {
//    int comm_size;
//    MPI_Comm_remote_size(communicator,&comm_size);
//    void* proposedAddr;
//    bool sucessMalloc=false;
//    while (!sucessMalloc) {
//        proposedAddr=getFreeVirtualAddr(size);
//        sucessMalloc=mmapOfAddr(proposedAddr);
//    }
//    //Fast mode: Try sending one random address and hope its free in every node
//    cacheOrder newOrder;
//    newOrder.opId=OPID_UNIFIED_MEM_REQ;
//    newOrder.size=size;
//    newOrder.hostAddr=(uint16_t)proposedAddr;
//    for (int i=0; i<comm_size ; i++) {
//       nanos::ext::MPIRemoteNode::nanosMPISend(&newOrder, 1, nanos::MPIDevice::cacheStruct, i, TAG_CACHE_ORDER, communicator);
//    }
//    int totalValids;
//    nanos::ext::MPIRemoteNode::nanosMPIRecv(&totalValids, 1, MPI_INT, 0, TAG_UNIFIED_MEM, communicator, MPI_STATUS_IGNORE);
//    if (totalValids!=comm_size) {
//        //Fast mode failed, enter safe mode
//        munmapOfAddr(proposedAddr);
//        uint64_t finalPtr;
//        uint64_t freeSpaces[N_FREE_SLOTS*2];
//        getFreeSpacesArr(freeSpaces,N_FREE_SLOTS*2);
//        MPI_Status status;
//        ptrArr[i]=freeSpaces;
//        arrLength[i]=N_FREE_SLOTS;
//        sizeArr[i]=freeSpaces+arrLength[i];
//        //Gather from everyone in the communicator
//        //MPI_Gather could be an option, but this is not a performance critical routine
//        //and would limit the sizes to a fixed size
//        for (int i=0; i<comm_size; ++i) {
//           MPI_Probe(parentRank, TAG_UNIFIED_MEM, parentcomm, &status);
            //TODO: FREE THIS MALLOCS
//           localArr= (uint64_t*) malloc(status.count*sizeof(uint64_t));
//           ptrArr[i]=localArr;
//           arrLength[i]=status.count/2;
//           sizeArr[i]=localArr+arrLength[i];
//           nanos::ext::MPIRemoteNode::nanosMPIRecv(&localArr, status.count, MPI_LONG, i, TAG_UNIFIED_MEM, communicator);
//        }
//        //Now intersect all the free spaces we got...
//        std::map<uint64_t,char> blackList;
//        bool sucess=false;
//        while (!sucess) {
//            finalPtr=getFreeChunk(comm_size+1, ptrArr, sizeArr, arrLength, order.size, blackList);
//            bool sucess=mmapOfAddr(finalPtr);
//            if (sucess) {
//                for (int i=0; i<comm_size ; i++) {
//                   nanos::ext::MPIRemoteNode::nanosMPISend(&finalPtr, 1, MPI_LONG, i, TAG_UNIFIED_MEM, communicator);
//                }
//                int totalValids;
//                nanos::ext::MPIRemoteNode::nanosMPIRecv(&totalValids, 1, MPI_INT, 0, TAG_UNIFIED_MEM, communicator, MPI_STATUS_IGNORE);
//                if (totalValids!=comm_size) {
//                    munmapOfAddr(finalPtr);
//                    sucess=false;
//                }
//            }
//            if (!sucess) blackList.insert(std::make_pair<uint64_t,char>(finalPtr,1));
//        }
//    }
}

void MPIRemoteNode::unifiedMemoryMallocRemote(cacheOrder& order, int parentRank, MPI_Comm parentcomm) {
//    int comm_size;
//    int localPositives=0;
//    int totalPositives=0;
//    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
//    bool hostProposalIsFree=true;
//    hostProposalIsFree=mmapOfAddr(order.hostAddr);
//    localPositives+=(int) hostProposalIsFree;
//    MPI_Allreduce(&localPositives, &totalPositives, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
//    if (rank==0) {
//        nanos::ext::MPIRemoteNode::nanosMPISend(&totalPositives, 1, MPI_INT, parentRank, TAG_UNIFIED_MEM, parentcomm);
//    }
//    if (totalPositives!=localPositives) {
//        //Some node failed doing malloc, entre safe mode
//        munmapOfAddr(order.hostAddr);
//        unifiedMemoryMallocRemoteSafe(order,parentRank,parentcomm);
//    }
}


void MPIRemoteNode::unifiedMemoryMallocRemoteSafe(cacheOrder& order, int parentRank, MPI_Comm parentcomm) {
//    uint64_t freeSpaces[N_FREE_SLOTS*2];
//    int comm_size;
//    int localPositives=0;
//    int totalPositives=-1;
//    //Send my array of free memory slots to the master
//    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
//    uint64_t finalPtr;
//    getFreeSpacesArr(freeSpaces,N_FREE_SLOTS*2);
//    nanos::ext::MPIRemoteNode::nanosMPISend(freespaces, sizeof(freeSpaces), MPI_LONG, parentRank, TAG_UNIFIED_MEM, parentcomm);
//    //Keep receiving addresses from the master until every node could malloc a common address
//    while (totalPositives!=localPositives) {
//        totalPositives=0;
//        localPositives=0;
//        //Wait for the answer of the proposed/valid pointer
//        nanos::ext::MPIRemoteNode::nanosMPIRecv(&finalPtr, 1, MPI_LONG, 0, TAG_UNIFIED_MEM, communicator, MPI_STATUS_IGNORE);
//
//        bool hostProposalIsFree=true;
//        hostProposalIsFree=mmapOfAddr(finalPtr);
//        localPositives+=(int) hostProposalIsFree;
//        MPI_Allreduce(&localPositives, &totalPositives, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
//        if (rank==0) {
//            nanos::ext::MPIRemoteNode::nanosMPISend(&totalPositives, 1, MPI_INT, parentRank, TAG_UNIFIED_MEM, parentcomm);
//        }
//        if (totalPositives!=localPositives) munmapOfAddr(finalPtr);
//    }
}

uint64_t MPIRemoteNode::getFreeChunk(int arraysLength, uint64_t** arrOfPtr,
         uint64_t** sizeArr,int** arrLength, size_t chunkSize, std::map<uint64_t,char>& blackList ) {
//    uint64_t result=0;
//    //TODO: Improve this implementation (brute force w/ many stop conditions + binary search right now)
//    //For every free chunk in each node, check if it's available in every other node
//    for (int masterSpaceNum=0; masterSpaceNum<arraysLength && resul!=0; ++masterSpaceNum) {
//        uint64_t* masterPtrArr=arrOfPtr[masterSpaceNum];
//        uint64_t* masterSizeArr=sizeArr[masterSpaceNum];
//        int masterArrLength=arrLength[masterSpaceNum];
//        for (int i=masterArrLength; i>=0 && resul!=0 ; --i) {
//            uint64_t masterPtr=masterPtrArr[i];
//            uint64_t masterSize=masterSizeArr[i];
//            if (masterSize>=chunkSize && blackList.count(masterPtr)==0) {
//                //Check if this pointer has free space
//                bool isAvaiableInAllSpaces=true;
//                for (int slaveSpaceNum=0; slaveSpaceNum<arraysLength && isAvaiableInAllSpaces; ++slaveSpaceNum) {
//                    bool spaceHasFreeChunk=false;
//                    uint64_t* slavePtrArr=arrOfPtr[slaveSpaceNum];
//                    uint64_t* slaveSizeArr=sizeArr[slaveSpaceNum];
//                    unsigned last=(unsigned)arrLength[slaveSpaceNum];
//                    unsigned min=0;
//                    unsigned mid=(min+last)/2;
//                    while (min <= last) {
//                        uint64_t slavePtr=slavePtrArr[mid];
//                        uint64_t slaveSize=slaveSizeArr[mid];
//
//                        if ( masterPtr>=slavePtr && masterPtr<=slavePtr+slaveSize)
//                        {
//                            //If there is space, mark it as free and stop, if not, just stop and discard this masterPtrv because the space
//                            //around it is not enough
//                            spaceHasFreeChunk= spaceHasFreeChunk || (masterPtr>=slavePtr &&
//                                    masterPtr+chunkSize <= slavePtr+slaveSize && blackList.count(slavePtr)==0);
//                            break;
//                        } else if ( slavePtr < masterPtr ) {
//                            first = mid+1;
//                        } else {
//                            last = mid-1;
//                        }
//                        mid= (first+last)>>1;
//                    }
//                    isAvaiableInAllSpaces= isAvaiableInAllSpaces && spaceHasFreeChunk;
//                }
//                if (isAvaiableInAllSpaces) {
//                    resul=masterPtr;
//                }
//            }
//        }
//    }
//    if (resul==0) {
//        fatal0("Couldn't find any free virtual address common to all nodes when trying to allocate unified memory space");
//    }
//    return result;
    return 0;
}


void MPIRemoteNode::DEEP_Booster_free(MPI_Comm *intercomm, int rank) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_DEEP_BOOSTER_FREE_EVENT);
    cacheOrder order;
    order.opId = OPID_FINISH;
    int nThreads=sys.getNumWorkers();
    //Now sleep the threads which represent the remote processes
    int res=MPI_IDENT;
    bool sharedSpawn=false;

    unsigned int numThreadsWithThisComm=0;
    std::vector<nanos::ext::MPIThread*> threadsToDespawn;
    std::vector<MPIProcessor*> communicatorsToFree;
    //Find threads and nodes to de-spawn
    for (int i=0; i< nThreads; ++i){
        BaseThread* bt=sys.getWorker(i);
        nanos::ext::MPIProcessor * myPE = dynamic_cast<nanos::ext::MPIProcessor *>(bt->runningOn());
        if (myPE && !bt->isSleeping()){
            MPI_Comm threadcomm=myPE->getCommunicator();
            if (threadcomm!=0 && intercomm!=NULL) MPI_Comm_compare(threadcomm,*intercomm,&res);
            if (res==MPI_IDENT){
                numThreadsWithThisComm++;
                if (myPE->getRank()==rank || rank == -1) {
                  sharedSpawn= sharedSpawn || myPE->getShared();
                  threadsToDespawn.push_back((nanos::ext::MPIThread *)bt);
                  //intercomm NULL (all communicators) and single rank is not supported (and it mostly makes no sense...)
                  //but it will work, except mpi comm free will have to be done by the user after he frees all ranks
                  if (intercomm==NULL && threadcomm != MPI_COMM_NULL && rank == -1) communicatorsToFree.push_back(myPE);
                }
            }
        }
    }
    if (!threadsToDespawn.empty()) {
        //All threads have the same commOfParents as they were spawned together
        MPI_Comm parentsComm=threadsToDespawn.front()->getRunningPEs().at(0)->getCommOfParents();
        int currRank=-1;
        MPI_Comm_rank(parentsComm,&currRank);
        //Synchronize parents before killing shared resources (as each parent only waits for his task
        //this prevents one parent killing a "son" which is still executing things from other parents)
        if (sharedSpawn) {
           MPI_Barrier(parentsComm);
        }
        //De-spawn threads and nodes
        for (std::vector<nanos::ext::MPIThread*>::iterator itThread = threadsToDespawn.begin(); itThread!=threadsToDespawn.end() ; ++itThread) {
            nanos::ext::MPIThread* mpiThread = *itThread;
            std::vector<MPIProcessor*>& myPEs = mpiThread->getRunningPEs();
            for (std::vector<MPIProcessor*>::iterator it = myPEs.begin(); it!=myPEs.end() ; ++it) {
                //Only owner will send kill signal to the worker
                if ( (*it)->getOwner() )
                {
                    nanosMPISend(&order, 1, nanos::MPIDevice::cacheStruct, (*it)->getRank(), TAG_M2S_ORDER, *intercomm);
                    //After sending finalization signals, we are not the owners anymore
                    //This way we prevent finalizing them multiple times if more than one thread uses them
                    (*it)->setOwner(false);
                }
                //If im the root do all the automatic control file work ONCE (first remote node), this time free the hosts
                if ( (*it)->getRank()==0 && currRank == 0 && !nanos::ext::MPIProcessor::getMpiControlFile().empty() ) {
                    //PPH List is the list of hosts which were consumed by the spawn of these ranks, lets "free" them
                    int* pph_list=(*it)->getPphList();
                    if (pph_list!=NULL) {
                        int fdAux=-1;
                        std::string controlName=nanos::ext::MPIProcessor::getMpiControlFile();
                        fdAux=tryGetLock(const_cast<char*> (controlName.c_str()));
                        if ( fdAux == -1 ) fatal0(controlName << " could not be opened/created, check your NX_OFFL_CONTROLFILE environment variable");
                        FILE* fd=fdopen(fdAux,"r+");
                        size_t num_bytes=0;
                        for (int i=0; pph_list[i] != -1 ; ++i) {
                            if (pph_list[i]!=0) {
                                fseek(fd, num_bytes, SEEK_SET);
                                const int freeNode=0;
                                fprintf(fd, "%d\n", freeNode);
                            }
                            num_bytes+=2;
                        }
                        fclose(fd);
                        delete[] pph_list;
                        (*it)->setPphList(NULL);
                        releaseLock(fdAux,const_cast<char*> (controlName.c_str()));
                    }
                }
            }
            if (rank==-1){
                mpiThread->lock();
                mpiThread->stop();
                mpiThread->join();
                mpiThread->unlock();
            }
        }
    }
    //If we despawned all the threads which used this communicator, free the communicator
    //If intercomm is null, do not do it, it should be the final free
    if (intercomm!=NULL
//            && threadsToDespawn.size()>=numThreadsWithThisComm
        ) {
/*
 * Uncomment when nanos offload and user's communicator handles are different
#ifdef OPEN_MPI
       MPI_Comm_free(intercomm);
#else
       MPI_Comm_disconnect(intercomm);
#endif
*/
        *intercomm = MPI_COMM_NULL;
    } else if (communicatorsToFree.size()>0) {
        for (std::vector<MPIProcessor*>::iterator it=communicatorsToFree.begin(); it!=communicatorsToFree.end(); ++it) {
           MPI_Comm comm = (*it)->getCommunicator();
#ifdef OPEN_MPI
           MPI_Comm_free( &comm );
#else
           MPI_Comm_disconnect( &comm );
#endif
           (*it)->setCommunicator( MPI_COMM_NULL );
        }
    }
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

void MPIRemoteNode::DEEPBoosterAlloc(MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm, bool strict, int* provided, int offset, int* pph_list) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_DEEP_BOOSTER_ALLOC_EVENT);
    //Initialize nanos MPI
    nanosMPIInit(0,0,MPI_THREAD_MULTIPLE,0);


    if (!MPIDD::getSpawnDone()) {
        int userProvided;
        MPI_Query_thread(&userProvided);
        if (userProvided < MPI_THREAD_MULTIPLE ) {
             message0("MPI_Query_Thread returned multithread support less than MPI_THREAD_MULTIPLE, your application may hang when offloading, check your MPI "
                "implementation and try to configure it so it can support this multithread level. Configure your PATH so the mpi compiler"
                " points to a multithread implementation of MPI");
             //Some implementations seem to catch fatal0 and continue... make sure we die
             exit(-1);
        }
    }

    std::vector<std::string> tokensParams;
    std::vector<std::string> tokensHost;
    std::vector<int> hostInstances;
    int totalNumberOfSpawns=0;
    int spawnedHosts=0;
    int rank;
    MPI_Comm_rank(comm,&rank);

    int availableHosts=0;
    //Read hostlist
    if ( rank == 0 && !nanos::ext::MPIProcessor::getMpiControlFile().empty() ) {
       buildHostLists(offset, INT_MAX, tokensParams, tokensHost, hostInstances); //Read all the hosts
       availableHosts=tokensHost.size();
    } else {
       buildHostLists(offset, number_of_hosts, tokensParams,tokensHost, hostInstances);
       availableHosts=tokensHost.size();
       if (availableHosts > number_of_hosts) availableHosts=number_of_hosts;
       //Check strict restrictions and react to them (return or set the right number of nodes)
       if (strict && number_of_hosts > availableHosts)
       {
          if (provided!=NULL) *provided=0;
          *intercomm=MPI_COMM_NULL;
          return;
       }
    }

    //If im the root do all the automatic control file work
    if ( rank == 0 && !nanos::ext::MPIProcessor::getMpiControlFile().empty() ) {
        if (offset != 0 || pph_list != NULL ) fatal0("NX_OFFL_CONTROLFILE environment variable has been defined, deep_booster_alloc_list is not supported"
                "/needed with automatic control of hosts");
        //Maximum length of pph_list alloc
        pph_list=new int[availableHosts+1];
        int fdAux=-1;
        int currStatus=0;
        std::string controlName=nanos::ext::MPIProcessor::getMpiControlFile();
        fdAux=tryGetLock(const_cast<char*> (controlName.c_str()));
        if ( fdAux == -1 ) fatal0(controlName << " could not be opened/created, check your NX_OFFL_CONTROLFILE environment variable");
        FILE* fd=fdopen(fdAux,"r+");
        int reserved=0;
        //Reserve as many hosts as needed
        int i;
        for (i=0;i < availableHosts && reserved < number_of_hosts; ++i) {
            long size=ftell(fd);
            int err=fscanf(fd, "%d\n", &currStatus);
            if (currStatus==0 || err == EOF ) {
                fseek(fd, size, SEEK_SET);
                pph_list[i]=process_per_host;
                const int busyNode=1;
                reserved++;
                fprintf(fd, "%d\n", busyNode);
            } else {
                pph_list[i]=0;
            }
        }
        fclose(fd);
        releaseLock(fdAux,const_cast<char*> (controlName.c_str()));
        //Mark the real length of pph_list
        availableHosts=i;
        pph_list[availableHosts]=-1; //end of list
        //Check strict restrictions and react to them (return or set the right number of nodes)
        if (strict && number_of_hosts > reserved)
        {
            if (provided!=NULL) *provided=0;
            *intercomm=MPI_COMM_NULL;
            return;
        }
    }

    //Register spawned processes so nanox can use them
    int mpiSize;
    MPI_Comm_size(comm,&mpiSize);
    bool shared=(mpiSize>1);

    callMPISpawn(comm, availableHosts, strict, tokensParams, tokensHost, hostInstances, pph_list,
            process_per_host, shared,/* outputs*/ spawnedHosts, totalNumberOfSpawns, intercomm);
    if (provided!=NULL) *provided=totalNumberOfSpawns;

    createNanoxStructures(comm, intercomm, spawnedHosts, totalNumberOfSpawns, shared, mpiSize, rank, pph_list);

    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}


static inline void trim(std::string& params){
    //Trim params
    size_t pos = params.find_last_not_of(" \t");
    if( std::string::npos != pos ) params = params.substr( 0, pos+1 );
    pos = params.find_first_not_of(" \t");
    if( std::string::npos != pos ) params = params.substr( pos );
}

void MPIRemoteNode::buildHostLists(
    int offset,
    int requestedHostNum,
    std::vector<std::string>& tokensParams,
    std::vector<std::string>& tokensHost,
    std::vector<int>& hostInstances)
{
    std::string mpiHosts=nanos::ext::MPIProcessor::getMpiHosts();
    std::string mpiHostsFile=nanos::ext::MPIProcessor::getMpiHostsFile();
    /** Build current host list */
    std::list<std::string> tmpStorage;
    //In case a host has no parameters, we'll fill our structure with this one
    std::string params="ompssnoparam";
    //Store single-line env value or hostfile into vector, separated by ';' or '\n'
    if ( !mpiHosts.empty() ){
        std::stringstream hostInput(mpiHosts);
        std::string line;
        while( getline( hostInput, line , ';') ){
            if (offset>0) offset--;
            else tmpStorage.push_back(line);
        }
    } else if ( !mpiHostsFile.empty() ){
        std::ifstream infile(mpiHostsFile.c_str());
        fatal_cond0(infile.bad(),"DEEP_Booster alloc error, NX_OFFL_HOSTFILE file not found");
        std::string line;
        while( getline( infile, line , '\n') ){
            if (offset>0) offset--;
            else tmpStorage.push_back(line);
        }
        infile.close();
    }

    while( !tmpStorage.empty() && (int)tokensHost.size() < requestedHostNum )
    {
        std::string line=tmpStorage.front();
        tmpStorage.pop_front();
        //If not commented add it to hosts
        if (!line.empty() && line.find("#")!=0){
            size_t posSep=line.find(":");
            size_t posEnd=line.find("<");
            if (posEnd==line.npos) {
                posEnd=line.size();
            } else {
                params=line.substr(posEnd+1,line.size());
                trim(params);
            }
            if (posSep!=line.npos){
                std::string realHost=line.substr(0,posSep);
                int number=atoi(line.substr(posSep+1,posEnd).c_str());
                trim(realHost);
                //Hosts with 0 instances in the file are ignored
                if (!realHost.empty() && number!=0) {
                    hostInstances.push_back(number);
                    tokensHost.push_back(realHost);
                    tokensParams.push_back(params);
                }
            } else {
                std::string realHost=line.substr(0,posEnd);
                trim(realHost);
                if (!realHost.empty()) {
                    hostInstances.push_back(1);
                    tokensHost.push_back(realHost);
                    tokensParams.push_back(params);
                }
            }
        }
    }
    bool emptyHosts=tokensHost.empty();
    //If there are no hosts, that means user "wants" to spawn in localhost
    while (emptyHosts && (int)tokensHost.size() < requestedHostNum ){
        tokensHost.push_back("localhost");
        tokensParams.push_back(params);
        hostInstances.push_back(1);
    }
    if (emptyHosts) {
        warning0("No hostfile or list was providen using NX_OFFL_HOSTFILE or NX_OFFL_HOSTS environment variables."
        " Deep_booster_alloc allocation will be performed in localhost (not recommended except for debugging)");
    }
}


void MPIRemoteNode::callMPISpawn(
    MPI_Comm comm,
    const int availableHosts,
    const bool strict,
    std::vector<std::string>& tokensParams,
    std::vector<std::string>& tokensHost,
    std::vector<int>& hostInstances,
    const int* pph_list,
    const int process_per_host,
    const bool& shared,
    int& spawnedHosts,
    int& totalNumberOfSpawns,
    MPI_Comm* intercomm)
{
    std::string mpiExecFile=nanos::ext::MPIProcessor::getMpiExecFile();
    std::string _mpiLauncherFile=nanos::ext::MPIProcessor::getMpiLauncherFile();
    bool pphFromHostfile=process_per_host<=0;
    bool usePPHList=pph_list!=NULL;
    // Spawn the remote process using previously parsed parameters
    std::string result_str;
    if ( !mpiExecFile.empty() ){
        result_str=mpiExecFile;
    } else {
        char result[ PATH_MAX ];
        ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
        std::string result_tmp(result, count);
        fatal_cond0(count==0,"Couldn't identify executable filename, please specify it manually using NX_OFFL_EXEC environment variable");
        result_str=result_tmp.substr(0,count);
    }

    /** Build spawn structures */
    //Number of spawns = max length (one instance per host)
    std::vector<char **> argvs(availableHosts);
    std::vector<char *> commands(availableHosts, &_mpiLauncherFile[0]);// Warning: This may not work
    std::vector<mpi::HostInfo> host_info(availableHosts);
    std::vector<int> num_processes(availableHosts,0);

    spawnedHosts=0;
    //This the real length of previously declared arrays, it will be equal to number_of_spawns when
    //hostfile/line only has one instance per host (aka no host:nInstances)
    for (int hostCounter=0; hostCounter < availableHosts; hostCounter++ ) {
        //Fill host
        std::string host;
        //Set number of instances this host can handle (depends if user specified, hostList specified or list specified)
        int currHostInstances;
        if (usePPHList) {
            currHostInstances=pph_list[hostCounter];
        } else if (pphFromHostfile){
            currHostInstances=hostInstances.at(hostCounter);
        } else {
            currHostInstances=process_per_host;
        }
        if (currHostInstances!=0) {
            host=tokensHost.at(hostCounter);
            //If host is a file, give it to Intel, otherwise put the host in the spawn
            std::ifstream hostfile(host.c_str());
            bool isfile=hostfile;
            if (isfile){
                std::string line;
                int number_of_lines_in_file=0;
                while (std::getline(hostfile, line)) {
                    ++number_of_lines_in_file;
                }

                host_info[hostCounter].set( "hostfile", host );
                currHostInstances=number_of_lines_in_file*currHostInstances;
            } else {
                host_info[hostCounter].set( "host", host );
            }
            //In case the MPI implementation supports soft key...
            if (!strict) {
                host_info[hostCounter].set( "soft", "0:N" );
            }

            hostfile.close();

            //Fill parameter array (including env vars)
            std::stringstream allParamTmp(tokensParams.at(hostCounter));
            std::string tmpParam;
            int paramsSize=3;
            while (getline(allParamTmp, tmpParam, ',')) {
                paramsSize++;
            }
            std::stringstream all_param(tokensParams.at(hostCounter));
            char **argvv=new char*[paramsSize];
            //Fill the params
            argvv[0]= const_cast<char*> (result_str.c_str());
            argvv[1]= const_cast<char*> ("empty");
            int paramCounter=2;
            while (getline(all_param, tmpParam, ',')) {
                //Trim current param
                trim(tmpParam);
                char* arg_copy=new char[tmpParam.size()+1];
                strcpy(arg_copy,tmpParam.c_str());
                argvv[paramCounter++]=arg_copy;
            }
            argvv[paramsSize-1]=NULL;

            commands[spawnedHosts]=const_cast<char*> (_mpiLauncherFile.c_str());
            argvs[spawnedHosts]=argvv;
            num_processes[spawnedHosts]=currHostInstances;

            totalNumberOfSpawns+=currHostInstances;
            ++spawnedHosts;
        }
    }
    #ifndef OPEN_MPI
    int fd=-1;
    //std::string lockname=NANOX_PREFIX"/bin/nanox-pfm";
    std::string lockname="./.ompssOffloadLock";
    while (!nanos::ext::MPIProcessor::isDisableSpawnLock() && !shared && fd==-1) {
       fd=tryGetLock(const_cast<char*> (lockname.c_str()));
    }
    #endif
    std::vector<MPI_Info> array_of_mpiinfo( host_info.begin(), host_info.end() );
    MPI_Comm_spawn_multiple(spawnedHosts,
				&commands.front(),
				&argvs.front(), &num_processes.front(),
            &array_of_mpiinfo.front(), 0, comm, intercomm, MPI_ERRCODES_IGNORE);
    #ifndef OPEN_MPI
    if (!nanos::ext::MPIProcessor::isDisableSpawnLock() && !shared) {
       releaseLock(fd,const_cast<char*> (lockname.c_str()));
    }
    #endif

    //Free all args sent
    for (int i=0;i<spawnedHosts;i++){
        //Free all args which were dynamically copied before
        for (int e=2;argvs[i][e]!=NULL;e++){
            delete[] argvs[i][e];
        }
        delete[] argvs[i];
    }
}


void MPIRemoteNode::createNanoxStructures(MPI_Comm comm, MPI_Comm* intercomm, int spawnedHosts, int totalNumberOfSpawns, bool shared, int mpiSize, int currRank, int* pphList){
    size_t maxWorkers= nanos::ext::MPIProcessor::getMaxWorkers();
    int spawn_start=0;
    int numberOfSpawnsThisProcess=totalNumberOfSpawns;
    //If shared (more than one parent for this group), split total spawns between nodes in order to balance syncs
    if (shared){
        numberOfSpawnsThisProcess=totalNumberOfSpawns/mpiSize;
        spawn_start=currRank*numberOfSpawnsThisProcess;
        if (currRank==mpiSize-1) //Last process syncs the remaining processes
            numberOfSpawnsThisProcess+=totalNumberOfSpawns%mpiSize;
    }

    PE* pes[totalNumberOfSpawns];
    //int uid=sys.getNumCreatedPEs();
    int arrSize;
    for (arrSize=0;ompss_mpi_masks[arrSize]==MASK_TASK_NUMBER;arrSize++){};
    int rank=spawn_start; //Balance spawn order so each process starts with his owned processes
    //Now they are spawned, send source ordering array so both master and workers have function pointers at the same position
    ext::SMPProcessor *core = sys.getSMPPlugin()->getLastFreeSMPProcessorAndReserve();
    if (core==NULL) {
        core = sys.getSMPPlugin()->getSMPProcessorByNUMAnode(0,getCurrentProcessor());
    }
    for ( int rankCounter=0; rankCounter<totalNumberOfSpawns; rankCounter++ ){
        memory_space_id_t id = sys.getNewSeparateMemoryAddressSpaceId();
        SeparateMemoryAddressSpace *mpiMem = NEW SeparateMemoryAddressSpace( id, nanos::ext::MPI, nanos::ext::MPIProcessor::getAllocWide());
        mpiMem->setNodeNumber( 0 );
        sys.addSeparateMemory(id,mpiMem);
        //Each process will have access to every remote node, but only one master will sync each child
        //this way we balance syncs with childs
        if (rank>=spawn_start && rank<spawn_start+numberOfSpawnsThisProcess) {
            pes[rank]=NEW nanos::ext::MPIProcessor( intercomm, rank,0/*uid++*/, true, shared, comm, core, id);
            ((MPIProcessor*)pes[rank])->setPphList(pphList);
            nanosMPISend(ompss_mpi_filenames, arrSize, MPI_UNSIGNED, rank, TAG_FP_NAME_SYNC, *intercomm);
            nanosMPISend(ompss_mpi_file_sizes, arrSize, MPI_UNSIGNED, rank, TAG_FP_SIZE_SYNC, *intercomm);
            //If user defined multithread cache behaviour, send the creation order
            if (nanos::ext::MPIProcessor::isUseMultiThread()) {
                cacheOrder order;
                order.opId = OPID_CREATEAUXTHREAD;
                nanosMPISend(&order, 1, nanos::MPIDevice::cacheStruct, rank, TAG_M2S_ORDER, *intercomm);
                ((MPIProcessor*)pes[rank])->setHasWorkerThread(true);
            }
        } else {
            pes[rank]=NEW nanos::ext::MPIProcessor( intercomm, rank,0/*uid++*/, false, shared, comm, core, id);
            ((MPIProcessor*)pes[rank])->setPphList(pphList);
        }
        rank=(rank+1)%totalNumberOfSpawns;
    }
    //Each node will have nSpawns/nNodes running, with a Maximum of 4
    //We supose that if 8 hosts spawns 16 nodes, each one will usually run 2
    //HINT: This does not mean that some remote nodes wont be accesible
    //using more than 1 thread is a performance tweak
    int numberOfThreads=(totalNumberOfSpawns/mpiSize);
    if (numberOfThreads<1) numberOfThreads=1;
    if (numberOfThreads>(int)maxWorkers) numberOfThreads=maxWorkers;
    BaseThread* threads[numberOfThreads];
    //start the threads...
    for (int i=0; i < numberOfThreads; ++i) {
        NANOS_INSTRUMENT( sys.getInstrumentation()->incrementMaxThreads(); )
        threads[i]=&((MPIProcessor*)pes[i])->startMPIThread(NULL);
    }
    sys.addPEsAndThreadsToTeam(pes, totalNumberOfSpawns, threads, numberOfThreads);
    nanos::ext::MPIPlugin::addPECount(totalNumberOfSpawns);
    nanos::ext::MPIPlugin::addWorkerCount(numberOfThreads);
    //Add all the PEs to the thread
    Lock* gLock=NULL;
    Atomic<unsigned int>* gCounter=NULL;
    std::vector<MPIThread*>* threadList=NULL;
    for ( spawnedHosts=0; spawnedHosts<numberOfThreads; spawnedHosts++ ){
        MPIThread* mpiThread=(MPIThread*) threads[spawnedHosts];
        //Get the lock of one of the threads
        if (gLock==NULL) {
            gLock=mpiThread->getSelfLock();
            gCounter=mpiThread->getSelfCounter();
            threadList=mpiThread->getSelfThreadList();
        }
        threadList->push_back(mpiThread);
        mpiThread->addRunningPEs((MPIProcessor**)pes,totalNumberOfSpawns);
        //Set the group lock so they all share the same lock
        mpiThread->setGroupCounter(gCounter);
        mpiThread->setGroupThreadList(threadList);
        if (numberOfThreads>1) {
            //Set the group lock so they all share the same lock
            mpiThread->setGroupLock(gLock);
        }
    }
    nanos::ext::MPIDD::setSpawnDone(true);
}

int MPIRemoteNode::nanosMPISendTaskinit(void *buf, int count, MPI_Datatype datatype, int dest,
        MPI_Comm comm) {
    cacheOrder order;
    order.opId = OPID_TASK_INIT;
    int* intbuf=(int*)buf;
    //Values of intbuf will be positive (using integer for conveniece with Fortran)
    order.hostAddr = *intbuf;
    //Send task init order (hostAddr=taskIdentifier)
    return nanosMPISend(&order, 1, nanos::MPIDevice::cacheStruct, dest, TAG_M2S_ORDER, comm);
}

int MPIRemoteNode::nanosMPIRecvTaskinit(void *buf, int count, MPI_Datatype datatype, int source,
        MPI_Comm comm, MPI_Status *status) {
    return nanosMPIRecv(buf, count, datatype, source, TAG_INI_TASK, comm, status);
}

int MPIRemoteNode::nanosMPISendTaskend(void *buf, int count, MPI_Datatype datatype, int disconnect,
        MPI_Comm comm) {
    if (_disconnectedFromParent) return 0;
    //Ignore destination (as is always parent) and get currentParent
    int res= nanosMPISend(buf, count, datatype, nanos::ext::MPIRemoteNode::getCurrentTaskParent(), TAG_END_TASK, comm);
    if (disconnect!=0) {
        MPI_Comm parent;
        MPI_Comm_get_parent(&parent);
        _disconnectedFromParent=true;
        MPI_Comm_disconnect(&parent);			
    }
    return res;
}

int MPIRemoteNode::nanosMPIRecvTaskend(void *buf, int count, MPI_Datatype datatype, int source,
        MPI_Comm comm, MPI_Status *status) {
    return nanosMPIRecv(buf, count, datatype, source, TAG_END_TASK, comm, status);
}

int MPIRemoteNode::nanosMPISendDatastruct(void *buf, int count, MPI_Datatype datatype, int dest,
        MPI_Comm comm) {
    return nanosMPISend(buf, count, datatype, dest, TAG_ENV_STRUCT, comm);
}

int MPIRemoteNode::nanosMPIRecvDatastruct(void *buf, int count, MPI_Datatype datatype, int source,
        MPI_Comm comm, MPI_Status *status) {
    //Ignore destination (as is always parent) and get currentParent
     nanosMPIRecv(buf, count, datatype,  nanos::ext::MPIRemoteNode::getCurrentTaskParent(), TAG_ENV_STRUCT, comm, status);
     return 0;
}

int MPIRemoteNode::nanosMPITypeCreateStruct( int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[],
        MPI_Datatype array_of_types[], MPI_Datatype **newtype, int taskId) {
    int err;
    *newtype= NEW MPI_Datatype;
    _taskStructsCache[taskId]=*newtype;
    err=MPI_Type_create_struct(count,array_of_blocklengths,array_of_displacements, array_of_types, *newtype );
    ensure0( err == MPI_SUCCESS, "MPI Create struct failed when preparing the task. Please submit a ticket" );
    err=MPI_Type_commit(*newtype);
    ensure0( err == MPI_SUCCESS, "MPI Create struct failed when preparing the task. Please submit a ticket" );
    return err;
}

void MPIRemoteNode::nanosMPITypeCacheGet( int taskId, MPI_Datatype **newtype ) {
    //Initialize cache if needed
    if (_taskStructsCache.size()==0) {
        //Fill total number of tasks which have been compiled
        int arr_size;
        for ( arr_size=0;ompss_mpi_masks[arr_size]==MASK_TASK_NUMBER;arr_size++ ){};
        unsigned int total_size=0;
        for ( int k=0;k<arr_size;k++ ) total_size+=ompss_mpi_file_ntasks[k];
        _taskStructsCache.assign(total_size, NULL);
    }
    ensure0 ( static_cast<int>(_taskStructsCache.size()) > taskId, "Tasks struct cache is failing, trying to access a taskId biggeer than the total number of tasks");
    *newtype=_taskStructsCache[taskId];
}

int MPIRemoteNode::nanosMPISend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
        MPI_Comm comm) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_SEND_EVENT);
    if (dest==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        dest=myPE->getRank();
        comm=myPE->getCommunicator();
    }
    //printf("Envio con tag %d, a %d\n",tag,dest);
    int err = MPI_Send(buf, count, datatype, dest, tag, comm);
    //printf("Fin Envio con tag %d, a %d\n",tag,dest);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return err;
}

int MPIRemoteNode::nanosMPIIsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
        MPI_Comm comm,MPI_Request *req) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_ISEND_EVENT);
    if (dest==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        dest=myPE->getRank();
        comm=myPE->getCommunicator();
    }
    //printf("Envio con tag %d, a %d\n",tag,dest);
    int err = MPI_Isend(buf, count, datatype, dest, tag, comm,req);
    //printf("Fin Envio con tag %d, a %d\n",tag,dest);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return err;
}

int MPIRemoteNode::nanosMPISsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
        MPI_Comm comm) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_SSEND_EVENT);
    if (dest==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        dest=myPE->getRank();
        comm=myPE->getCommunicator();
    }
    //printf("Enviobloq con tag %d, a %d\n",tag,dest);
    int err = MPI_Ssend(buf, count, datatype, dest, tag, comm);
    //printf("Fin Enviobloq con tag %d, a %d\n",tag,dest);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return err;
}

int MPIRemoteNode::nanosMPIRecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
        MPI_Comm comm, MPI_Status *status) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RECV_EVENT);
    if (source==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        source=myPE->getRank();
        comm=myPE->getCommunicator();
    }
    //printf("recv con tag %d, desde %d\n",tag,source);
    int err = MPI_Recv(buf, count, datatype, source, tag, comm, status );
    //printf("Fin recv con tag %d, desde %d\n",tag,source);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return err;
}

int MPIRemoteNode::nanosMPIIRecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
        MPI_Comm comm, MPI_Request *req) {
    NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_IRECV_EVENT);
    if (source==UNKOWN_RANKSRCDST){
        nanos::ext::MPIProcessor * myPE = ( nanos::ext::MPIProcessor * ) myThread->runningOn();
        source=myPE->getRank();
        comm=myPE->getCommunicator();
    }
    //printf("recv con tag %d, desde %d\n",tag,source);
    int err = MPI_Irecv(buf, count, datatype, source, tag, comm, req );
    //printf("Fin recv con tag %d, desde %d\n",tag,source);
    NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
    return err;
}

/**
 * Synchronizes host and device function pointer arrays to ensure that are in the same order
 * in both files (host and device, which are different architectures, so maybe they were not compiled in the same order)
 */
void MPIRemoteNode::nanosSyncDevPointers(int* file_mask, unsigned int* file_namehash, unsigned int* file_size,
            unsigned int* task_per_file,void *ompss_mpi_func_ptrs_dev[]){
    MPI_Comm parentcomm; /* intercommunicator */
    MPI_Comm_get_parent(&parentcomm);
    //If this process was not spawned, we don't need this reorder (and shouldnt have been called)
    if ( parentcomm != 0 && parentcomm != MPI_COMM_NULL ) {
        //MPI_Status status;
        int arr_size;
        for ( arr_size=0;file_mask[arr_size]==MASK_TASK_NUMBER;arr_size++ ){};
        unsigned int total_size=0;
        for ( int k=0;k<arr_size;k++ ) {
           //Files which have 0 tasks, may add a NULL to the pointer array
           //if this is the case their number of tasks for reordering purposes is 1
           size_t element_offset=total_size*sizeof(void*);
           //All these many transformations are used to avoid warnings on gcc
           if (task_per_file[k]==0 &&  ((void*) *(void**)(((uint64_t)ompss_mpi_func_ptrs_dev)+element_offset))==NULL ) task_per_file[k]=1;
           total_size+=task_per_file[k];
        }

        size_t filled_arr_size=0;
        unsigned int* host_file_size=(unsigned int*) malloc(sizeof(unsigned int)*arr_size);
        unsigned int* host_file_namehash=(unsigned int*) malloc(sizeof(unsigned int)*arr_size);
        void (**ompss_mpi_func_pointers_dev_out)()=(void (**)()) malloc(sizeof(void (*)())*total_size);
        //Receive host information
        nanos::ext::MPIRemoteNode::nanosMPIRecv(host_file_namehash, arr_size, MPI_UNSIGNED, MPI_ANY_SOURCE, TAG_FP_NAME_SYNC, parentcomm, MPI_STATUS_IGNORE);
        nanos::ext::MPIRemoteNode::nanosMPIRecv(host_file_size, arr_size, MPI_UNSIGNED, MPI_ANY_SOURCE, TAG_FP_SIZE_SYNC, parentcomm, MPI_STATUS_IGNORE );
        int i,e,func_pointers_arr;
        bool found;
        //i loops at host files
        for ( i=0;i<arr_size;i++ ){
            func_pointers_arr=0;
            found=false;
            //Search the host file in dev file and copy every pointer in the same order
            for ( e=0;!found && e<arr_size;e++ ){
                if( file_namehash[e] == host_file_namehash[i] && file_size[e] == host_file_size[i] ){
                    found=true;
                    //Copy from _dev_tmp array to _dev array in the same order than the host
                    memcpy(ompss_mpi_func_pointers_dev_out+filled_arr_size,ompss_mpi_func_ptrs_dev+func_pointers_arr,task_per_file[e]*sizeof(void (*)()));
                    filled_arr_size+=task_per_file[e];
                }
                func_pointers_arr+=task_per_file[e];
            }
            fatal_cond0(!found,"Executable version mismatch, please compile the code using exactly the same sources (same filename, mod. date and size) for every architecture");
        }
        memcpy(ompss_mpi_func_ptrs_dev,ompss_mpi_func_pointers_dev_out,total_size*sizeof(void (*)()));
        free(ompss_mpi_func_pointers_dev_out);
        free(host_file_size);
        free(host_file_namehash);
    }
}
