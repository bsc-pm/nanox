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

#ifndef _NANOS_GPU_PROCESSOR
#define _NANOS_GPU_PROCESSOR


#include "accelerator.hpp"
#include "cache.hpp"
#include "config.hpp"
#include "gpudevice.hpp"
#include "gputhread.hpp"


namespace nanos {
namespace ext
{

struct GPUInfo {
   size_t maxMemoryAvailable;
};

   class GPUProcessor : public Accelerator
   {
      public:

         class GPUInfo
         {
            private:
               size_t _maxMemoryAvailable;

            public:
               GPUInfo ( int device );

               size_t getMaxMemoryAvailable () { return _maxMemoryAvailable; }
         };

      private:

         // configuration variables
         static Atomic<int>      _deviceSeed; // Number of GPU devices assigned to threads
         int                     _gpuDevice; // Assigned GPU device Id
         GPUInfo                 _gpuInfo; // Information related to the GPU device that represents

         // cache
         DeviceCache<GPUDevice>  _cache;

         // disable copy constructor and assignment operator
         GPUProcessor( const GPUProcessor &pe );
         const GPUProcessor & operator= ( const GPUProcessor &pe );

         size_t getMaxMemoryAvailable ( int id );

      public:
         // constructors
         GPUProcessor( int id, int gpuId ) : Accelerator( id, &GPU ), _gpuDevice( _deviceSeed++ ), _gpuInfo( gpuId ), _cache()
         {
            std::cout << "[GPUProcessor] I have " << _gpuInfo.getMaxMemoryAvailable()
                  << " bytes of available memory (device #" << gpuId << ")" << std::endl;
         }

         virtual ~GPUProcessor() {}

         virtual WD & getWorkerWD () const;
         virtual WD & getMasterWD () const;
         virtual BaseThread & createThread ( WorkDescriptor &wd );

         // capability query functions
         virtual bool supportsUserLevelThreads () const { return false; }

         /* Memory space support */
         virtual void registerCacheAccessDependent( uint64_t tag, size_t size, bool input, bool output );
         virtual void unregisterCacheAccessDependent( uint64_t tag, size_t size );
         virtual void registerPrivateAccessDependent( uint64_t tag, size_t size, bool input, bool output );
         virtual void unregisterPrivateAccessDependent( uint64_t tag, size_t size );

         virtual void* getAddressDependent( uint64_t tag );
         virtual void copyToDependent( void *dst, uint64_t tag, size_t size );
   };

}
}

#endif
