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


#include "config.hpp"
#include "gputhread.hpp"
#include "cache.hpp"
#include "gpudevice.hpp"
#include "accelerator.hpp"


namespace nanos {
namespace ext
{

   class GPUProcessor : public Accelerator
   {


      private:
         // config variables
         static bool _useUserThreads;
         static size_t _threadsStackSize;

         // disable copy constructor and assignment operator
         GPUProcessor( const GPUProcessor &pe );
         const GPUProcessor & operator= ( const GPUProcessor &pe );

         Cache<GPUDevice> _cache;

      public:
         // constructors
         GPUProcessor( int id ) : Accelerator( id, &GPU ), _cache() {}

         virtual ~GPUProcessor() {}

         virtual WD & getWorkerWD () const;
         virtual WD & getMasterWD () const;
         virtual BaseThread & createThread ( WorkDescriptor &wd );

         static void prepareConfig ( Config &config );
         // capability query functions
         virtual bool supportsUserLevelThreads () const { return false; }

         /* Memory space support */
         virtual void registerDataAccessDependent( uint64_t tag, size_t size );
         virtual void copyDataDependent( uint64_t tag, size_t size );
         virtual void unregisterDataAccessDependent( uint64_t tag );
         virtual void copyBackDependent( uint64_t tag, size_t size );
         virtual void* getAddressDependent( uint64_t tag );
         virtual void copyToDependent( void *dst, uint64_t tag, size_t size );
   };

}
}

#endif
