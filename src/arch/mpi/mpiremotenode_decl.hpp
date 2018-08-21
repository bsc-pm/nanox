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

#ifndef _NANOS_MPI_REMOTE_NODE_DECL
#define _NANOS_MPI_REMOTE_NODE_DECL

#include "atomic_decl.hpp"

#include "config.hpp"

#include "mpidevice.hpp"
#include "mpithread.hpp"

#include "concurrent_queue.hpp"
#include "commanddispatcher.hpp"

#include "mpispawn_fwd.hpp"

#include <map>
#include <mpi.h>

extern "C" {

extern __attribute__((weak)) int ompss_mpi_masks[];
extern __attribute__((weak)) unsigned int ompss_mpi_filenames[];
extern __attribute__((weak)) unsigned int ompss_mpi_file_sizes[];
extern __attribute__((weak)) unsigned int ompss_mpi_file_ntasks[];
extern __attribute__((weak)) void *ompss_mpi_func_pointers_host[];
extern __attribute__((weak)) void *ompss_mpi_func_pointers_dev[];

}

namespace nanos {
namespace ext {

        // Probably better to use unordered_map
        typedef std::map< MPI_Comm, mpi::RemoteSpawn* > RemoteSpawnMap;

        /**
         * Class which implements all the remote-node logic which is not in the cache (executing tasks etc...)
         * Also implements user-APIs like deep_booster/free calls, nanos mpi free/inits and all the wrappers for MPI send/recvs.
         * Has access to MPIProcessor private variables because user configuration options are kept there
         */
        class MPIRemoteNode {
        private:
            static size_t _bufferDefaultSize;
            static char* _bufferPtr;
            
            static bool _initialized;   
            static bool _disconnectedFromParent;

            static ProducerConsumerQueue<std::pair<int,int> >* _pendingTasksWithParent;
            static mpi::command::Dispatcher* _commandDispatcher;
            static std::vector<MPI_Datatype*> _taskStructsCache;   

            static int _currentTaskParent;
            static int _currProcessor;

            static RemoteSpawnMap _spawnedRemotes;

            // disable copy constructor and assignment operator
            MPIRemoteNode(const MPIRemoteNode &pe);
            const MPIRemoteNode & operator=(const MPIRemoteNode &pe);

        public:
            static int getCurrentTaskParent();
            
            static void setCurrentTaskParent(int parent);
            
            static int getCurrentProcessor();

            static std::pair<int,int> getNextTaskAndParent();

            static bool isNextTaskAvailable();

            static void addTaskToQueue(int task_id, int parentId);

            static bool getDisconnectedFromParent();            
            
            static bool executeTask(int taskId);

            static mpi::command::Dispatcher& getDispatcher();

            static void registerSpawn( MPI_Comm communicator, mpi::RemoteSpawn& spawn );

            static RemoteSpawnMap& getRegisteredSpawns();
            
            /**
             * Initialize OmpSs
             * (Nanox MPI Init & slave sync offload structures with master)
             */
            static void preInit(); 
            
            /**
             * Offload main's (receive tasks and exit)
             */
            static void mpiOffloadSlaveMain();
            
            //Search function pointer and get index of host array
            static int ompssMpiGetFunctionIndexHost(void* func_pointer);
            
            //Search function pointer and get index of device array
            static int ompssMpiGetFunctionIndexDevice(void* func_pointer);
            
            /**
             * This routine implements a worker thread which will execute
             * tasks whenever notified by the cache thread
             * @return 0
             */
            static int nanosMPIWorker();     
            
            
            /**
            * Statics (mostly external API adapters provided to user or used by mercurium) begin here
            */
            
            static void nanosMPIInit(int* argc, char ***argv, int required, int* provided);
            
            static void nanosMPIFinalize();
            
            /**
             * API called by the user which starts a malloc of unified mem address between hosts
             * and everyone in this communicator
             * @param size
             * @param communicator
             */
            static void unifiedMemoryMallocHost(size_t size, MPI_Comm communicator);

            /**
             * Intersect memory spaces and get a chunk of chunkSize
             * @param arraysLength Length of the arrays (num mem spaces)
             * @param ptrArr array of free spaces (ordered)
             * @param sizeArr array of free size after each ptrArr
             * @param arrLength array of lengths of the spaces and spaces
             * @param chunkSize size of the free space to search
             * @param blackList pointers which should not be considered (failed before)
             * @return 
             */
            static uint64_t getFreeChunk(int arraysLength, uint64_t** arrOfPtrs,
                             uint64_t** sizeArr,int** arrLength, size_t chunkSize, std::map<uint64_t,char>& blackList );

            
            /**
             * Free booster group
             * @param intercomm Communicator (NULL means free every booster registered)
             * @param rank rank
             */
            static void DEEP_Booster_free(MPI_Comm *intercomm, int rank);
            
            /**
             * Function which allocates remote nodes
             * Composed of thee parts
             * 1- Build host list (aka read host list from file/env var and in a future job manager)
             * 2- Perform the call to MPI spawn
             * 3- Build nanox structures and threads
             * @param comm Communicator of the masters who will allocate (collective) and access the boosters
             * @param number_of_hosts Number of hosts (lines in hostfile) to spawn
             * @param process_per_host Process per host to spawn
             * @param intercomm Resulting interccomm representing the boosters
             * @param strict Boolean which indicates if the call should crash when not enough nodes are avaiable
             * @param provided [OUT] returns the real number of hosts allocated (only makes sense with strict=false)
             * @param offset Offset (0 by default, only used in deep_booster_alloc_offset)
             * @param id_host_list List individually which specifies which hosts will be used (of length number_of_hosts)
             * @param pph_list Process per host when using id_host_list (of length number_of_hosts)
             * Process per host will be 0 (aka incompatible) with the "_list" mode
             * Offset will be 0 (aka incompatible) with the "_list" mode
             */
            static void DEEPBoosterAlloc(MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm,
                    bool strict, int* provided,
                    int offset, int* pph_list);  
            
            /*
             * Subkernel for DEEPBoosterAlloc
            */
            static inline void buildHostLists( 
                int offset,
                int requested_host_num,
                std::vector<std::string>& tokens_params,
                std::vector<std::string>& tokens_host, 
                std::vector<int>& host_instances);
            
            /*
             * Subkernel for DEEPBoosterAlloc
            */
            static inline void callMPISpawn( 
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
                MPI_Comm* intercomm);
            
            
            /*
             * Subkernel for DEEPBoosterAlloc
            */
            static inline void createNanoxStructures( 
                MPI_Comm comm,
                MPI_Comm* intercomm,
                int spawnedHosts,
                int totalNumberOfSpawns,
                bool shared,
                int mpiSize,
                int currRank,
                int* pphList);
            
            /**
             * Wrappers for MPI functions
             */
            
            /**
             * Send task init signal
             * @param buf buffer pointing to an integer
             * @param count 1
             * @param datatype IGNORED
             * @param dest
             * @param comm
             * @return 
             */
            static int nanosMPISendTaskInit(void *buf, int count, int dest, MPI_Comm comm);

            static int nanosMPISendTaskEnd(void *buf, int count, MPI_Datatype datatype, int disconnect,
                    MPI_Comm comm);

            static int nanosMPIRecvTaskEnd(void *buf, int count, MPI_Datatype datatype, int source,
                    MPI_Comm comm, MPI_Status *status);

            static int nanosMPISendDatastruct(void *buf, int count, MPI_Datatype datatype, int dest,
                    MPI_Comm comm);

            static int nanosMPIRecvDatastruct(void *buf, int count, MPI_Datatype datatype, int source,
                    MPI_Comm comm, MPI_Status *status);

            static int nanosMPISend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm);
            
            static int nanosMPISsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm);
            
            static int nanosMPIIsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm,MPI_Request *req);

            static int nanosMPIRecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                    MPI_Comm comm, MPI_Status *status);
            
            static int nanosMPIIRecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                     MPI_Comm comm, MPI_Request *req);
            
            static int nanosMPITypeCreateStruct(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], 
                    MPI_Datatype array_of_types[], MPI_Datatype **newtype, int taskId);
            
            static void nanosMPITypeCacheGet( int taskId, MPI_Datatype **newtype );
                        
                        
            /**
             * Specialized functions
             */
            static void nanosSyncDevPointers(int* file_mask, unsigned int* file_namehash, unsigned int* file_size,
                    unsigned int* task_per_file,void* ompss_mpi_func_pointers_dev[]);
            
        };   

} // namespace ext
} // namespace nanos

#endif
