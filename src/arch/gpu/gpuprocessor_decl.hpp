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

#ifndef _NANOS_GPU_PROCESSOR_DECL
#define _NANOS_GPU_PROCESSOR_DECL

#include "cachedaccelerator.hpp"
#include "gputhread_decl.hpp"
#include "gpuconfig.hpp"
#include "gpudevice_decl.hpp"
#include "gpumemorytransfer_decl.hpp"
#include "gpuutils.hpp"
#include "gpumemoryspace_fwd.hpp"
#include "malign.hpp"
#include "simpleallocator_decl.hpp"
#include "copydescriptor_decl.hpp"
#include "smpprocessor.hpp"

#include <map>


namespace nanos {
namespace ext {

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

   class GPUProcessor : public ProcessingElement
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



      private:
         // Configuration variables
         static Atomic<int>      _deviceSeed; //! Number of GPU devices assigned to threads
         int                     _gpuDevice; //! Assigned GPU device Id
         //size_t                  _memoryAlignment;
         GPUProcessorTransfers   _gpuProcessorTransfers; //! Keep the list of pending memory transfers
         GPUProcessorInfo *      _gpuProcessorInfo; //! Information related to the GPU device that represents
         GPUProcessorStats       _gpuProcessorStats; //! Statistics of data copied in and out to / from cache
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
         bool                    _initialized; //! Object is initialized
#else
         volatile bool           _initialized; //! Object is initialized
#endif
         GPUMemorySpace         &_gpuMemory;
         SMPProcessor           *_core;
         BaseThread             *_thread;


         //SimpleAllocator               _allocator;

         //! Disable copy constructor and assignment operator
         GPUProcessor( const GPUProcessor &pe );
         const GPUProcessor & operator= ( const GPUProcessor &pe );


      public:
         //! Constructors
         GPUProcessor( int gpuId, memory_space_id_t memId, SMPProcessor *core, GPUMemorySpace &gpuMem );

         virtual ~GPUProcessor();

         void init();
         void cleanUp();
         void freeWholeMemory();
         GPUMemorySpace &getGPUMemory() { return _gpuMemory; }

         WD & getWorkerWD () const;
         WD & getMasterWD () const;
         virtual WD & getMultiWorkerWD () const
         {
            fatal( "getMultiWorkerWD: GPUProcessor is not allowed to create MultiThreads" );
         }

         BaseThread & createThread ( WorkDescriptor &wd, SMPMultiThread *parent );

         virtual BaseThread & createMultiThread ( WorkDescriptor &wd, unsigned int numPEs, PE **repPEs )
         {
            fatal( "GPUProcessor is not allowed to create MultiThreads" );
         }

         //! Capability query functions
         bool supportsUserLevelThreads () const { return false; }

         int getDeviceId ()
         {
            return _gpuDevice;
         }

         // Allocator interface
         //void * allocate ( size_t size )
         //{
         //   return _allocator.allocate( NANOS_ALIGNED_MEMORY_OFFSET( 0, size, _memoryAlignment ) );
         //}

         //void free( void * address )
         //{
         //   _allocator.free( address );
         //}

         GPUMemoryTransferList * getOutTransferList ();
         GPUMemoryTransferList * getInTransferList ();

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


         void printStats ();

         void setInitialized ()
         {
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
            __atomic_store_n(&_initialized, true, __ATOMIC_RELEASE);
#else
            _initialized = true;
            memoryFence();
#endif
         }
         void waitInitialized ()
         {
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
            while ( ! __atomic_load_n(&_initialized, __ATOMIC_ACQUIRE) ) { }
#else
            while ( !_initialized ) { memoryFence(); }
#endif
         }
         std::size_t getMaxMemoryAvailable () const;

      // Methods related to GPUThread management
      protected:
         ProcessingElement::ThreadList & getThreads() { return _core->getThreads(); }

      public:
         BaseThread & startGPUThread();

         std::size_t getNumThreads() const { return _core->getNumThreads(); }
         void stopAllThreads ();
         BaseThread * getFirstThread();
//xteruel
#if 0
         BaseThread * getFirstRunningThread_FIXME();
         BaseThread * getFirstStoppedThread_FIXME();
         BaseThread * getUnassignedThread();
#endif

   };

} // namespace ext
} // namespace nanos

#endif
