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

#ifndef _NANOS_GPU_PROCESSOR_DECL
#define _NANOS_GPU_PROCESSOR_DECL

#include "cachedaccelerator.hpp"
#include "gputhread_decl.hpp"
#include "gpuconfig.hpp"
#include "gpudevice_decl.hpp"
#include "gpumemorytransfer_decl.hpp"
#include "gpuutils.hpp"
#include "malign.hpp"
#include "simpleallocator_decl.hpp"
#include "copydescriptor_decl.hpp"

#include <map>


namespace nanos {
namespace ext
{

   class GPUProcessor : public CachedAccelerator<GPUDevice>
   {
      public:
         class GPUProcessorInfo;

         class GPUProcessorStats
         {
            public:
               Atomic<unsigned int>    _bytesIn;
               Atomic<unsigned int>    _bytesOut;
               Atomic<unsigned int>    _bytesDevice;

               GPUProcessorStats()
               {
                  _bytesIn = 0;
                  _bytesOut = 0;
                  _bytesDevice = 0;
               }
         };

         class GPUProcessorTransfers
         {
            public:
               GPUMemoryTransferList * _pendingCopiesIn;
               GPUMemoryTransferList * _pendingCopiesOut;


               GPUProcessorTransfers()
               {
                  _pendingCopiesIn = NEW GPUMemoryTransferInAsyncList();
                  _pendingCopiesOut = NEW GPUMemoryTransferOutSyncList();
               }

               ~GPUProcessorTransfers() 
               {
                  delete _pendingCopiesIn;
                  delete _pendingCopiesOut;
               }
         };


      private:
         // Configuration variables
         static Atomic<int>      _deviceSeed; //! Number of GPU devices assigned to threads
         int                     _gpuDevice; //! Assigned GPU device Id
         size_t                  _memoryAlignment;
         GPUProcessorInfo *      _gpuProcessorInfo; //! Information related to the GPU device that represents
         GPUProcessorStats       _gpuProcessorStats; //! Statistics of data copied in and out to / from cache
         GPUProcessorTransfers   _gpuProcessorTransfers; //! Keep the list of pending memory transfers


         SimpleAllocator               _allocator;
         BufferManager                 _inputPinnedMemoryBuffer;
         BufferManager                 _outputPinnedMemoryBuffer;

         //! Disable copy constructor and assignment operator
         GPUProcessor( const GPUProcessor &pe );
         const GPUProcessor & operator= ( const GPUProcessor &pe );

         size_t getMaxMemoryAvailable ( int id );

      public:
         //! Constructors
         GPUProcessor( int id, int gpuId, int uid );

         virtual ~GPUProcessor();

         void init();
         void cleanUp();
         void freeWholeMemory();

         WD & getWorkerWD () const;
         WD & getMasterWD () const;
         BaseThread & createThread ( WorkDescriptor &wd );

         //! Capability query functions
         bool supportsUserLevelThreads () const { return false; }

         int getDeviceId ()
         {
            return _gpuDevice;
         }

         // Allocator interface
         void * allocate ( size_t size )
         {
            return _allocator.allocate( NANOS_ALIGNED_MEMORY_OFFSET( 0, size, _memoryAlignment ) );
         }

         void free( void * address )
         {
            _allocator.free( address );
         }

         void * allocateInputPinnedMemory ( size_t size )
         {
            return _inputPinnedMemoryBuffer.allocate( size );
         }

         void freeInputPinnedMemory ()
         {
            _inputPinnedMemoryBuffer.reset();
         }

         void * allocateOutputPinnedMemory ( size_t size )
         {
            return _outputPinnedMemoryBuffer.allocate( size );
         }

         void freeOutputPinnedMemory ()
         {
            _outputPinnedMemoryBuffer.reset();
         }

         //! Get information about the GPU that represents this object
         GPUProcessorInfo * getGPUProcessorInfo ()
         {
            return _gpuProcessorInfo;
         }

         void transferInput ( size_t size )
         {
            _gpuProcessorStats._bytesIn += ( unsigned int ) size;
         }

         void transferOutput ( size_t size )
         {
            _gpuProcessorStats._bytesOut += ( unsigned int ) size;
         }

         void transferDevice ( size_t size )
         {
            _gpuProcessorStats._bytesDevice += ( unsigned int ) size;
         }

         GPUMemoryTransferList * getInTransferList ()
         {
            return _gpuProcessorTransfers._pendingCopiesIn;
         }

         GPUMemoryTransferList * getOutTransferList ()
         {
            return _gpuProcessorTransfers._pendingCopiesOut;
         }

         void printStats ()
         {
            message("GPU " << _gpuDevice << " TRANSFER STATISTICS");
            message("    Total input transfers: " << bytesToHumanReadable( _gpuProcessorStats._bytesIn.value() ) );
            message("    Total output transfers: " << bytesToHumanReadable( _gpuProcessorStats._bytesOut.value() ) );
            message("    Total device transfers: " << bytesToHumanReadable( _gpuProcessorStats._bytesDevice.value() ) );
         }
   };

}
}

#endif
