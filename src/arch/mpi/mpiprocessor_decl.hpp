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
#include "mpiremotenode_decl.hpp"

namespace nanos {
namespace ext {

        class MPIProcessor : public ProcessingElement {
        private:
            // config variables
            static bool _useUserThreads;
            static size_t _threadsStackSize;
            
            //MPI Node data
            static size_t _cacheDefaultSize;
            static System::CachePolicyType _cachePolicy;
            //! Save OmpSS-mpi filename
            static std::string _mpiExecFile;
            static std::string _mpiLauncherFile;
            static std::string _mpiHosts;
            static std::string _mpiHostsFile;    
            static std::string _mpiControlFile;   
            static bool _useMultiThread;
            static bool _allocWide;
            #ifndef OPEN_MPI
            static bool _disableSpawnLock;  
            #endif
            static int _numPrevPEs;
            static int _numFreeCores;
            static int _currPE;
            static size_t _alignThreshold;          
            static size_t _alignment;          
            static size_t _maxWorkers;
            
            MPI_Comm _communicator;
            int _rank;
            bool _owner; //if we are the owner (process in charge of freeing the remote process)
            bool _shared; //if more processes also have access to this PE
            bool _hasWorkerThread;
            int* _pphList; //saves which hosts in list/hostfile were ocuppied by this spawn

            Atomic<bool> _busy;
            WorkDescriptor* _currExecutingWd;
            int _currExecutingDD;
            std::list<MPI_Request> _pendingReqs;
            MPI_Comm _commOfParents;

            SMPProcessor* _core;
            Lock _peLock;
            

            // disable copy constructor and assignment operator
            MPIProcessor(const MPIProcessor &pe);
            const MPIProcessor & operator=(const MPIProcessor &pe);


        public:
            
            //MPIProcessor( int id ) : PE( id, &MPI ) {}
            MPIProcessor( void* communicator, int rank, int uid, bool owned, bool shared, MPI_Comm commOfParents, SMPProcessor* core, memory_space_id_t memId );

            ~MPIProcessor() {                
            }            

            /* Nanox NX_OFFL  Configuration options*/
            static size_t getCacheDefaultSize();

            static System::CachePolicyType getCachePolicy();
            static std::string getMpiHosts();
            
            static std::string getMpiHostsFile();
            
            static std::string getMpiControlFile();
            
            static std::string getMpiExecFile();

            static std::string getMpiLauncherFile();
            
            static size_t getAlignment();
            
            static size_t getAlignThreshold();            


            #ifndef OPEN_MPI
            static bool isDisableSpawnLock();
            #endif
                        
            static bool getAllocWide();

            static size_t getMaxWorkers();

            static bool isUseMultiThread();
            /* End config options*/           
            
            

            MPI_Comm getCommunicator() const;
            
            MPI_Comm getCommOfParents() const;     
 
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
            
            void setPphList(int* list);
            
            int* getPphList();
                      
            /**
             * Try to reserve this PE, if the one who reserves it is the same
            */
            bool testAndSetBusy(int dduid, bool multithreadedAccess);
            
            int getCurrExecutingDD() const;

            void setCurrExecutingDD(int currExecutingDD);
            
            void appendToPendingRequests(MPI_Request& req);
            
            /**
             * Waits or clears all Pending Requests
             * Non thread-safe function
             */
            void clearAllRequests();
            /**
             * Tests all pending requests
             * Thread-safe function
             */
            bool testAllRequests();

            BaseThread& startMPIThread(WD* work);
            
            WD & getWorkerWD() const;
            WD & getMasterWD() const;
            static void prepareConfig(Config &config);

            // capability query functions
            virtual WD & getMultiWorkerWD () const
            {
               fatal( "getMultiWorkerWD: GPUProcessor is not allowed to create MultiThreads" );
            }
            BaseThread & createThread ( WorkDescriptor &wd, SMPMultiThread *parent );
            
            virtual BaseThread & createMultiThread ( WorkDescriptor &wd, unsigned int numPEs, PE **repPEs )
            {
               fatal( "ClusterNode is not allowed to create MultiThreads" );
            }

            bool supportsUserLevelThreads() const {
                return false;
            }       
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
           NANOS_MPI_IRECV_EVENT,                     /* 24 */
           NANOS_MPI_ISEND_EVENT,                     /* 25 */
           NANOS_MPI_GENERIC_EVENT                         /* 26 */
        } in_mpi_runtime_event_value;

} // namespace ext
} // namespace nanos
#endif
