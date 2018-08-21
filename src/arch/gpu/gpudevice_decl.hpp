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

#ifndef _GPU_DEVICE_DECL
#define _GPU_DEVICE_DECL

#include "workdescriptor.hpp"
#include "processingelement_fwd.hpp"
#include "copydescriptor_decl.hpp"
#include "gpuprocessor_fwd.hpp"

namespace nanos {

/* \brief Device specialization for GPU architecture
 * provides functions to allocate and copy data in the device
 */
   class GPUDevice : public Device
   {
      private:
         static unsigned int _rlimit;

         static void getMemoryLockLimit();

         /*! \brief copy in when the thread invoking this function belongs to pe
          */
         void isMycopyIn( void *localDst, CopyDescriptor &remoteSrc, size_t len, size_t count, size_t ld, SeparateMemoryAddressSpace &mem, ext::GPUProcessor *gpu );


         /*! \brief copy in when the thread invoking this function does not belong to pe
          *         In this case, the information about the copy is added to a list, and the appropriate
          *         thread (which is periodically checking the list) will perform the copy and notify
          *         the cache when it has finished
          */
         void isNotMycopyIn( void *localDst, CopyDescriptor &remoteSrc, size_t len, size_t count, size_t ld, SeparateMemoryAddressSpace &mem, ext::GPUProcessor *gpu );


         /*! \brief copy out when the thread invoking this function belongs to pe
          */
         void isMycopyOut( CopyDescriptor &remoteDst, void *localSrc, size_t len, size_t count, size_t ld, SeparateMemoryAddressSpace &mem, ext::GPUProcessor *gpu );


         /*! \brief copy out when the thread invoking this function does not belong to pe
          *         In this case, the information about the copy is added to a list, and the appropriate
          *         thread (which is periodically checking the list) will perform the copy and notify
          *         the cache when it has finished
          */
         void isNotMycopyOut( CopyDescriptor &remoteDst, void *localSrc, size_t len, size_t count, size_t ld, SeparateMemoryAddressSpace &mem, ext::GPUProcessor *gpu );


         /*! \brief GPUDevice copy constructor
          */
         explicit GPUDevice ( const GPUDevice &arch );

      public:
         /*! \brief GPUDevice constructor
          */
         GPUDevice ( const char *n );

         /*! \brief GPUDevice destructor
          */
         ~GPUDevice();

         /* \brief allocate the whole memory of the GPU device
          *        If the allocation fails due to a CUDA memory-related error,
          *        this function keeps trying to allocate as much memory as
          *        possible by trying smaller sizes from 100% to 50%, decrementing
          *        by 5% each time
          *        On success, returns a pointer to the allocated memory and rewrites
          *        size with the final amount of allocated memory
          */
         static void * allocateWholeMemory( size_t &size );

         /* \brief free the whole GPU device memory pointed by address
          */
         static void freeWholeMemory( void * address );

         /* \brief allocate a chunk of pinned memory of the host
          */
         static void * allocatePinnedMemory( size_t size );
         static void * allocatePinnedMemory2( size_t size );

         /* \brief free the chunk of pinned host memory pointed by address
          */
         static void freePinnedMemory( void * address );

         /* \brief allocate size bytes in the device
          */
         static void * allocate( size_t size, ProcessingElement *pe, uint64_t tag = 0 );

         /* \brief free address
          */
         static void free( void *address, ProcessingElement *pe );

         /* \brief Copy from remoteSrc in the host to localDst in the device
          *        This operation can either be synchronous or asynchronous
          */
         virtual void _copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops,
               WD const *wd, void *hostObject, reg_t hostRegionId ) ;

         /* \brief Copy from localSrc in the device to remoteDst in the host
          *        This operation can either be synchronous or asynchronous
          */
         virtual void _copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops,
               WD const *wd, void *hostObject, reg_t hostRegionId ) ;

         /* \brief Copy locally in the device from src to dst
          */
         static void copyLocal( void *dst, void *src, size_t size, ProcessingElement *pe );

         /* \brief When using asynchronous transfer modes, this function is used to notify
          *        the PE that another GPU has requested the data synchronization related to
          *        hostAddress
          */
         static void syncTransfer( uint64_t hostAddress, ProcessingElement *pe);

         /* \brief Reallocate and copy from address
          */
         static void * realloc( void * address, size_t size, size_t ceSize, ProcessingElement *pe );

         /* \brief copy from src in the host to dst in the device synchronously
          */
         static void copyInSyncToDevice ( void * dst, void * src, size_t len, size_t count, size_t ld );

         /* \brief when transferring with asynchronous modes, copy from src in the host
          *        to dst in the host, where dst is an intermediate buffer
          */
         static void copyInAsyncToBuffer( void * dst, void * src, size_t len, size_t count, size_t ld );

         /* \brief when transferring with asynchronous modes, copy from src in the host
          *        to dst in the device, where src is an intermediate buffer
          */
         static void copyInAsyncToDevice( void * dst, void * src, size_t len, size_t count, size_t ld );

         /* \brief when transferring with asynchronous modes, wait until all input copies
          *        (from host to device) have been completed
          */
         static void copyInAsyncWait();

         /* \brief when transferring with synchronous mode, copy from src in the device
          *        to dst in the host
          */
         static void copyOutSyncToHost ( void * dst, void * src, size_t len, size_t count, size_t ld );

         /* \brief when transferring with asynchronous modes, copy from src in the device
          *        to dst in the host, where dst is an intermediate buffer
          */
         static void copyOutAsyncToBuffer( void * src, void * dst, size_t len, size_t count, size_t ld );

         /* \brief when transferring with asynchronous modes, wait until all output copies
          *        (from device to host) have been completed
          */
         static void copyOutAsyncWait();

         /* \brief when transferring with asynchronous modes, copy from src in the host
          *        to dst in the host, where src is an intermediate buffer
          */
         static void copyOutAsyncToHost( void * src, void * dst, size_t len, size_t count, size_t ld );

         /* \brief Copy from addrSrc in peSrc device to addrDst in peDst device
          *        Returns true if the operation is synchronous
          */
         virtual bool _copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest,
               SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );

         virtual void *memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, WD const *wd, unsigned int copyIdx );

         virtual void memFree( uint64_t addr, SeparateMemoryAddressSpace &mem );

         virtual std::size_t getMemCapacity( SeparateMemoryAddressSpace &mem );

         virtual void _canAllocate( SeparateMemoryAddressSpace &mem, std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes );

         virtual void _getFreeMemoryChunksList( SeparateMemoryAddressSpace &mem, SimpleAllocator::ChunkList &list );

         virtual void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t count, std::size_t ld,
               SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );

         virtual void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t count, std::size_t ld,
               SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );

         virtual bool _copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t count,
               std::size_t ld, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops,
               WD const *wd, void *hostObject, reg_t hostRegionId );

         void syncTransfer( uint64_t hostAddress, SeparateMemoryAddressSpace &mem, ext::GPUProcessor *gpu ) const;
   };

} // namespace nanos

#endif
