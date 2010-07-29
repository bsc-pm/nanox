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

#include "accelerator.hpp"
#include "cache.hpp"
#include "config.hpp"
#include "gpudevice.hpp"
#include "gputhread.hpp"

#include <map>


namespace nanos {
namespace ext
{

   class GPUProcessor : public Accelerator
   {
      public:
         class GPUProcessorInfo;

      private:
         // Configuration variables
         static Atomic<int>      _deviceSeed; // Number of GPU devices assigned to threads
         int                     _gpuDevice; // Assigned GPU device Id
         GPUProcessorInfo *      _gpuProcessorInfo; // Information related to the GPU device that represents

         // Cache
         DeviceCache<GPUDevice>        _cache;
         std::map< void *, uint64_t >  _pinnedMemory;

         // Disable copy constructor and assignment operator
         GPUProcessor( const GPUProcessor &pe );
         const GPUProcessor & operator= ( const GPUProcessor &pe );

         size_t getMaxMemoryAvailable ( int id );

      public:
         // Constructors
         GPUProcessor( int id, int gpuId );

         virtual ~GPUProcessor() {}

         virtual WD & getWorkerWD () const;
         virtual WD & getMasterWD () const;
         virtual BaseThread & createThread ( WorkDescriptor &wd );

         // Capability query functions
         virtual bool supportsUserLevelThreads () const { return false; }

         // Memory space support
         virtual void registerCacheAccessDependent( uint64_t tag, size_t size, bool input, bool output );
         virtual void unregisterCacheAccessDependent( uint64_t tag, size_t size );
         virtual void registerPrivateAccessDependent( uint64_t tag, size_t size, bool input, bool output );
         virtual void unregisterPrivateAccessDependent( uint64_t tag, size_t size );

         virtual void* getAddressDependent( uint64_t tag );
         virtual void copyToDependent( void *dst, uint64_t tag, size_t size );

         // Get information about the GPU that represents this object
         GPUProcessorInfo * getGPUProcessorInfo ()
         {
            return _gpuProcessorInfo;
         }

         uint64_t getPinnedAddress ( void * dAddress )
         {
            return _pinnedMemory[dAddress];
         }

         void setPinnedAddress ( void * dAddress, uint64_t pinned )
         {
            _pinnedMemory[dAddress] = pinned;
         }

         void removePinnedAddress ( void * dAddress )
         {
            _pinnedMemory.erase( dAddress );
         }
   };

}
}

#endif
