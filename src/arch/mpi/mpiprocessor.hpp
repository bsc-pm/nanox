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

            // disable copy constructor and assignment operator
            MPIProcessor(const MPIProcessor &pe);
            const MPIProcessor & operator=(const MPIProcessor &pe);


        public:
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
            MPI_Comm _communicator;
            int _rank;
            
            //MPIProcessor( int id ) : PE( id, &MPI ) {}
            MPIProcessor(int id, MPI_Comm communicator, int rank);

            virtual ~MPIProcessor() {
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
            
            static void nanos_MPI_Init(int* argc, char ***argv);
            
            static void nanos_MPI_Finalize();
            
            static void DEEP_Booster_alloc(MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm);  
            
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

            static int nanos_MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                    MPI_Comm comm, MPI_Status *status);
        };

    }
}

#endif
