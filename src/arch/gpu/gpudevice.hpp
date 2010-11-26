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

#ifndef _GPU_DEVICE
#define _GPU_DEVICE

#include "workdescriptor_decl.hpp"
#include "processingelement_fwd.hpp"
#include "copydescriptor_decl.hpp"

namespace nanos
{

   typedef enum {
      NANOS_GPU_TRANSFER_NORMAL,         //!  Basic transfer mode with no overlap
      NANOS_GPU_TRANSFER_ASYNC,          //! -- A little bit better (gives bad results from time to time)
      NANOS_GPU_TRANSFER_PINNED_CUDA,    //! -- Slowdown of ~10x (gives always bad results)
      NANOS_GPU_TRANSFER_PINNED_OS,      //! -- Similar to NANOS_GPU_TRANSFER_NORMAL (correct results though mlock fails)
      NANOS_GPU_TRANSFER_WC              //! -- Same as NANOS_GPU_TRANSFER_PINNED_CUDA: Slowdown of ~10x (gives always bad results)
   } transfer_mode;

/* \brief Device specialization for GPU architecture
 * provides functions to allocate and copy data in the device
 */

   class GPUDevice : public Device
   {
      private:
         static transfer_mode _transferMode;

         static unsigned int _rlimit;

         static void getMemoryLockLimit();

         /*! \brief GPUDevice copy constructor
          */
         explicit GPUDevice ( const GPUDevice &arch );

      public:
         /*! \brief GPUDevice constructor
          */
         GPUDevice ( const char *n ) : Device ( n )
         {
            getMemoryLockLimit();
         }

         /*! \brief GPUDevice destructor
          */
         ~GPUDevice() {};

         /* \brief choose the transfer mode for GPU devices
          */
         static void setTransferMode ( transfer_mode mode )
         {
            _transferMode = mode;
         }

         /* \brief get the transfer mode for GPU devices
          */
         static transfer_mode getTransferMode ()
         {
            return _transferMode;
         }

         /* \brief allocate the whole memory of the GPU device
          *        If the allocation fails due to a CUDA memory-related error,
          *        this function keeps trying to allocate as much memory as
          *        possible by trying smaller sizes from 100% to 50%, decrementing
          *        by 5% each time
          *        On success, returns a pointer to the allocated memory and rewrites
          *        size with the final amount of allocated memory 
          */
         static void * allocateWholeMemory( size_t &size );

         /* \brief free the whole GPU device memory pointed by address */
         static void freeWholeMemory( void * address );

         /* \brief allocate size bytes in the device */
         static void * allocate( size_t size, ProcessingElement *pe );

         /* \brief free address */
         static void free( void *address, ProcessingElement *pe );

         /* \brief Copy from remoteSrc in the host to localDst in the device
          *        Returns true if the operation is synchronous
          */
         static bool copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe );

         /* \brief Copy from localSrc in the device to remoteDst in the host
          *        Returns true if the operation is synchronous
          */
         static bool copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe );

         /* \brief Copy locally in the device from src to dst
          */
         static void copyLocal( void *dst, void *src, size_t size, ProcessingElement *pe );

         /* \brief When using asynchronous transfer modes, this function is used to notify
          *        the PE that another GPU has requested the data synchronization related to 
          *        hostAddress
          */
         static void syncTransfer( uint64_t hostAddress, ProcessingElement *pe);

         /* \brief Reallocate and copy from address.  */
         static void * realloc( void * address, size_t size, size_t ceSize, ProcessingElement *pe );

         /* \brief when transferring with asynchronous modes, copy from src in the device
          *        to dst in the host, where dst is an intermediate buffer */
         static void copyOutAsyncToBuffer( void * src, void * dst, size_t size );

         /* \brief when transferring with asynchronous modes, wait until all output copies
          *        (from device to host) have been completed
          *        dst is an intermediate buffer
          */
         static void copyOutAsyncWait();

         /* \brief when transferring with asynchronous modes, copy from src in the host
          *        to dst in the host, where src is an intermediate buffer
          */
         static void copyOutAsyncToHost( void * src, void * dst, size_t size );

         /* \brief when transferring with synchronous mode, copy from src in the device
          *        to dst in the host
          */
         static void copyOutSyncToHost ( void * dst, void * src, size_t size );


   };
}

#endif
