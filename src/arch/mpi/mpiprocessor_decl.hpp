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

#ifndef _NANOS_MPI_PROCESSOR_DECL
#define _NANOS_MPI_PROCESSOR_DECL

#include "mpi.h"
#include "atomic_decl.hpp"
#include "config.hpp"
#include "mpidevice.hpp"
#include "mpithread.hpp"
#include "cachedaccelerator.hpp"
#include "copydescriptor_decl.hpp"
#include "processingelement.hpp"

namespace nanos {
    namespace ext {

        class MPIProcessor : public CachedAccelerator<MPIDevice> {
        private:
            // config variables
            static bool _useUserThreads;
            static size_t _threadsStackSize;
            static size_t _bufferDefaultSize;
            static char* _bufferPtr;
            
            //MPI Node data
            static size_t _cacheDefaultSize;
            static System::CachePolicyType _cachePolicy;
            //! Save OmpSS-mpi filename
            static std::string _mpiExecFile;
            static std::string _mpiLauncherFile;
            static std::string _mpiHosts;
            static std::string _mpiHostsFile;    
            static bool _useMultiThread;  
            static int _numPrevPEs;
            static int _numFreeCores;
            static int _currPE;
            static bool _inicialized;
            static int _currentTaskParent;      
            static size_t _alignThreshold;          
            static size_t _alignment;          
            static size_t _maxWorkers;
            
            
            static Lock _taskLock;
            static int _currTaskIdentifier;
            
            
            MPI_Comm _communicator;
            int _rank;
            bool _owner;
            bool _shared;
            bool _hasWorkerThread;
            Atomic<bool> _busy;
            WorkDescriptor* _currExecutingWd;
            int _currExecutingDD;
            std::vector<MPI_Request> _pendingReqs;
            

            // disable copy constructor and assignment operator
            MPIProcessor(const MPIProcessor &pe);
            const MPIProcessor & operator=(const MPIProcessor &pe);


        public:
            
            //MPIProcessor( int id ) : PE( id, &MPI ) {}
            MPIProcessor(int id, void* communicator, int rank, int uid, bool owned, bool shared);

            ~MPIProcessor() {                
            }
            
            static size_t getCacheDefaultSize();

            static System::CachePolicyType getCachePolicy();

            MPI_Comm getCommunicator() const;

            static std::string getMpiFilename();

            static std::string getMpiHosts();
            
            static std::string getMpiHostsFile();

            static std::string getMpiLauncherFile();
            
            static size_t getAlignment();
            
            static size_t getAlignThreshold();
                         
            static int getCurrentTaskParent();
            
            static Lock& getTaskLock();
            
            static int getCurrTaskIdentifier();
            
            static void setCurrTaskIdentifier(int val);

            static void setCurrentTaskParent(int parentId);
 
            int getRank() const;
            
            bool getOwner() const;
            
            void setOwner(bool owner);
            
            bool getHasWorkerThread() const;
            
            void setHasWorkerThread(bool hwt);
            
            bool getShared() const;

            WD* getCurrExecutingWd() const;

            void setCurrExecutingWd(WD* currExecutingWd);

            bool isBusy() const;

            void setBusy(bool busy);
            
            //Try to reserve this PE, if the one who reserves it is the same
            //which already has the PE, return true
            bool testAndSetBusy(int dduid);
            
            int getCurrExecutingDD() const;

            void setCurrExecutingDD(int currExecutingDD);
            
            void appendToPendingRequests(MPI_Request& req);
            
            void clearAllRequests();

            
            WD & getWorkerWD() const;
            WD & getMasterWD() const;
            BaseThread & createThread(WorkDescriptor &wd);
            static void prepareConfig(Config &config);

            // capability query functions

            bool supportsUserLevelThreads() const {
                return false;
            }       
            
            
            
            static bool executeTask(int taskId);
            
            /**
             * This routine implements a worker thread which will execute
             * tasks whenever notified by the cache thread
             * @return 0
             */
            static int nanosMPIWorker();        
            
            static int getNextPEId();    
            
            
            /**
            * Statics (mostly external API adapters provided to user or used by mercurium) begin here
            */

            static void DEEP_Booster_free(MPI_Comm *intercomm, int rank);
            
            static void nanosMPIInit(int* argc, char ***argv, int required, int* provided);
            
            static void nanosMPIFinalize();
                        
            static void DEEPBoosterAlloc(MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm, int offset);  
            
            static int nanosMPISendTaskinit(void *buf, int count, MPI_Datatype datatype, int dest,
                    MPI_Comm comm);

            static int nanosMPIRecvTaskinit(void *buf, int count, MPI_Datatype datatype, int source,
                    MPI_Comm comm, MPI_Status *status); 

            static int nanosMPISendTaskend(void *buf, int count, MPI_Datatype datatype, int dest,
                    MPI_Comm comm);

            static int nanosMPIRecvTaskend(void *buf, int count, MPI_Datatype datatype, int source,
                    MPI_Comm comm, MPI_Status *status);

            static int nanosMPISendDatastruct(void *buf, int count, MPI_Datatype datatype, int dest,
                    MPI_Comm comm);

            static int nanosMPIRecvDatastruct(void *buf, int count, MPI_Datatype datatype, int source,
                    MPI_Comm comm, MPI_Status *status);

            static int nanosMPISend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm);
            
            static int nanosMPISsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm);
            
            static int nanosMPIIsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
             MPI_Comm comm,MPI_Request *req);

            static int nanosMPIRecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                    MPI_Comm comm, MPI_Status *status);
            
            static int nanosMPITypeCreateStruct(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], 
                    MPI_Datatype array_of_types[], MPI_Datatype *newtype);
            
            static void nanosSyncDevPointers(int* file_mask, unsigned int* file_namehash, unsigned int* file_size,
                    unsigned int* task_per_file,void (*ompss_mpi_func_pointers_dev[])());
            
            static void mpiOffloadSlaveMain();
            
            //Search function pointer and get index
            static int ompssMpiGetFunctionIndexHost(void* func_pointer);
            
        };   

        // Macros to instrument the code and make it cleaner
        #define NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(x)   NANOS_INSTRUMENT( \
              sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "in-mpi-runtime" ), (x) ); )

        #define NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT       NANOS_INSTRUMENT( \
              sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "in-mpi-runtime" ), 0 ); )


        typedef enum {
           NANOS_MPI_NULL_EVENT,                            /* 0 */
           NANOS_MPI_ALLOC_EVENT,                          /* 1 */
           NANOS_MPI_FREE_EVENT,                            /* 2 */
           NANOS_MPI_DEEP_BOOSTER_ALLOC_EVENT,                     /* 3 */
           NANOS_MPI_COPYIN_SYNC_EVENT,                         /* 4 */
           NANOS_MPI_COPYOUT_SYNC_EVENT,                 /* 5 */
           NANOS_MPI_COPYDEV2DEV_SYNC_EVENT,                 /* 6 */
           NANOS_MPI_DEEP_BOOSTER_FREE_EVENT,                     /* 7 */
           NANOS_MPI_INIT_EVENT,                     /* 8 */
           NANOS_MPI_FINALIZE_EVENT,                     /* 9 */
           NANOS_MPI_SEND_EVENT,                     /* 10 */
           NANOS_MPI_RECV_EVENT,                     /* 11 */
           NANOS_MPI_SSEND_EVENT,                     /* 12 */
           NANOS_MPI_COPYLOCAL_SYNC_EVENT,                     /* 13 */
           NANOS_MPI_REALLOC_EVENT,                     /* 14 */
           NANOS_MPI_WAIT_FOR_COPIES_EVENT,                     /* 15 */
           NANOS_MPI_RNODE_COPYIN_EVENT,                     /* 16 */
           NANOS_MPI_RNODE_COPYOUT_EVENT,                     /* 17 */
           NANOS_MPI_RNODE_DEV2DEV_IN_EVENT,                     /* 18 */
           NANOS_MPI_RNODE_DEV2DEV_OUT_EVENT,                     /* 19 */
           NANOS_MPI_RNODE_ALLOC_EVENT,                     /* 20 */
           NANOS_MPI_RNODE_REALLOC_EVENT,                     /* 21 */
           NANOS_MPI_RNODE_FREE_EVENT,                     /* 22 */
           NANOS_MPI_RNODE_COPYLOCAL_EVENT,                     /* 23 */
           NANOS_MPI_GENERIC_EVENT                         /* 24 */
        } in_mpi_runtime_event_value;

    }

}
#endif
