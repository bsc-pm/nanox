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

#ifndef _NANOS_MPI_PROCESSOR
#define _NANOS_MPI_PROCESSOR

#include "mpi.h"
#include "atomic_decl.hpp"
#include "config.hpp"
#include "mpidevice.hpp"
#include "mpithread.hpp"
#include "cachedaccelerator.hpp"
#include "copydescriptor_decl.hpp"
#include "processingelement.hpp"

//This var must keep same value than in mercurium MPI Device
#define TAG_MAIN_OMPSS "__ompss_mpi_daemon" 

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
            static std::string _mpiFilename;
            static std::string _mpiHosts;
            static std::string _mpiHostsFile;
            static int _mpiFileArrSize;
            static unsigned int* _mpiFileHashname;
            static unsigned int* _mpiFileSize;            
            static int _numPrevPEs;
            static int _numFreeCores;
            static int _currPE;
            MPI_Comm _communicator;
            int _rank;

            // disable copy constructor and assignment operator
            MPIProcessor(const MPIProcessor &pe);
            const MPIProcessor & operator=(const MPIProcessor &pe);


        public:
            
            //MPIProcessor( int id ) : PE( id, &MPI ) {}
            MPIProcessor(int id, void* communicator, int rank, int uid);

            virtual ~MPIProcessor() {
            }
            
            static size_t getCacheDefaultSize() {
                return _cacheDefaultSize;
            }

            static System::CachePolicyType getCachePolicy() {
                return _cachePolicy;
            }

            MPI_Comm getCommunicator() const {
                return _communicator;
            }

            static int getMpiFileArrSize() {
                return _mpiFileArrSize;
            }

            static void setMpiFileArrSize(int mpiFileArrSize) {
                _mpiFileArrSize = mpiFileArrSize;
            }

            static unsigned int* getMpiFileHashname() {
                return _mpiFileHashname;
            }

            static void setMpiFileHashname(unsigned int* mpiFileHashname) {
                _mpiFileHashname = mpiFileHashname;
            }

            static unsigned int* getMpiFileSize() {
                return _mpiFileSize;
            }

            static void setMpiFileSize(unsigned int* mpiFileSize) {
                _mpiFileSize = mpiFileSize;
            }

            static std::string getMpiFilename() {
                return _mpiFilename;
            }

            static std::string getMpiHosts() {
                return _mpiHosts;
            }
            static std::string getMpiHostsFile() {
                return _mpiHostsFile;
            }

            static std::string getMpiLauncherFile() {
                return _mpiLauncherFile;
            }
 
            int getRank() const {
                return _rank;
            }


            virtual WD & getWorkerWD() const;
            virtual WD & getMasterWD() const;
            virtual BaseThread & createThread(WorkDescriptor &wd);

            static void prepareConfig(Config &config);

            static void setMpiExename(char* new_name);

            static std::string getMpiExename();

            static void DEEP_Booster_free(MPI_Comm *intercomm, int rank);

            // capability query functions

            virtual bool supportsUserLevelThreads() const {
                return false;
            }

            /**
             * Nanos MPI override
             **/                        
            
            static int getNextPEId();
            
            static void nanos_MPI_Init(int* argc, char ***argv);
            
            static void nanos_MPI_Finalize();
            
            static void DEEP_Booster_alloc(MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm, int offset);  
            
            static int nanos_MPI_Send_taskinit(void *buf, int count, MPI_Datatype datatype, int dest,
                    MPI_Comm comm);

            static int nanos_MPI_Recv_taskinit(void *buf, int count, MPI_Datatype datatype, int source,
                    MPI_Comm comm, MPI_Status *status); 

            static int nanos_MPI_Send_taskend(void *buf, int count, MPI_Datatype datatype, int dest,
                    MPI_Comm comm);

            static int nanos_MPI_Recv_taskend(void *buf, int count, MPI_Datatype datatype, int source,
                    MPI_Comm comm, MPI_Status *status);

            static int nanos_MPI_Send_datastruct(void *buf, int count, MPI_Datatype datatype, int dest,
                    MPI_Comm comm);

            static int nanos_MPI_Recv_datastruct(void *buf, int count, MPI_Datatype datatype, int source,
                    MPI_Comm comm, MPI_Status *status);

            static int nanos_MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm);
            
            static int nanos_MPI_Ssend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm);
            
            static int nanos_MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
             MPI_Comm comm,MPI_Request *req);

            static int nanos_MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                    MPI_Comm comm, MPI_Status *status);
            
            static int nanos_MPI_Type_create_struct(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], 
                    MPI_Datatype array_of_types[], MPI_Datatype *newtype);
            
            
        };   

        // Macro's to instrument the code and make it cleaner
        #define NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(x)   NANOS_INSTRUMENT( \
              sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "in-mpi-runtime" ), (x) ); )

        #define NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT       NANOS_INSTRUMENT( \
              sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "in-mpi-runtime" ) ); )


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
