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

#ifndef _MPI_DEVICE
#define _MPI_DEVICE

#include "workdescriptor_decl.hpp"
#include "processingelement_fwd.hpp"
#include "commandid.hpp"

#include <mpi.h>

namespace nanos {

/* \brief Device specialization for MPI architecture
 * provides functions to allocate and copy data in the device
 */
   class MPIDevice : public Device
   {
      private:
         static char _executingTask;
         //static Directory *_masterDir;
         static bool _createdExtraWorkerThread;

         static void getMemoryLockLimit();

         /*! \brief MPIDevice copy constructor
          */
         explicit MPIDevice ( const MPIDevice &arch );

      public:

         static MPI_Datatype cacheStruct;
         /*! \brief MPIDevice constructor
          */
         MPIDevice ( const char *n );

         /*! \brief MPIDevice destructor
          */
         ~MPIDevice();

         /* \brief Reallocate and copy from address
          */
         static void * realloc( void * address, size_t size, size_t ceSize, ProcessingElement *pe );

         /**
          * Initialice cache struct datatype MPI so we can use it
          */
         static void initMPICacheStruct();

         template < bool dedicated >
         static void remoteNodeCacheWorker();

         static void createExtraCacheThread();

         virtual void *memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, WorkDescriptor const *wd, unsigned int copyIdx);
         virtual void memFree( uint64_t addr, SeparateMemoryAddressSpace &mem );
         virtual void _canAllocate( SeparateMemoryAddressSpace &mem, std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes ) { }
         virtual std::size_t getMemCapacity( SeparateMemoryAddressSpace &mem );

         virtual void _copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
         virtual void _copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
         virtual bool _copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
         virtual void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) { fatal0("Strided copies not implemented if offload");}
         virtual void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) { fatal0("Strided copies not implemented if offload");}
         virtual bool _copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) { fatal0("Strided copies not implemented if offload"); }
         virtual void _getFreeMemoryChunksList( SeparateMemoryAddressSpace &mem, SimpleAllocator::ChunkList &list ) { fatal0(__PRETTY_FUNCTION__ << " not implemented if offload"); }

   };

} // namespace nanos

#endif // _MPI_DEVICE
