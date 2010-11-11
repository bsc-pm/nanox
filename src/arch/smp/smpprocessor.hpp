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

#ifndef _NANOS_SMP_PROCESSOR
#define _NANOS_SMP_PROCESSOR

#include "config.hpp"
#include "smpthread.hpp"
#include "smpdevice.hpp"
#ifdef SMP_NUMA
#include "cache.hpp"
#include "accelerator.hpp"
#else
#include "processingelement.hpp"
#endif

//TODO: Make smp independent from pthreads? move it to OS?

namespace nanos {
namespace ext
{

#ifdef SMP_NUMA
   class SMPProcessor : public Accelerator
#else
   class SMPProcessor : public PE
#endif
   {


      private:
         // config variables
         static bool _useUserThreads;
         static size_t _threadsStackSize;
         static size_t _cacheDefaultSize;

         // disable copy constructor and assignment operator
         SMPProcessor( const SMPProcessor &pe );
         const SMPProcessor & operator= ( const SMPProcessor &pe );

#ifdef SMP_NUMA
#ifdef CLUSTER_DEV
         DeviceCache<SMPDevice> _cache;
         Lock _lock;
#else
         DeviceCache<SMPDevice> _cache;
         Lock _lock;
#endif
#endif

      public:
         // constructors
#ifdef SMP_NUMA
         SMPProcessor( int id ) : Accelerator( id, &SMP ), _cache(_cacheDefaultSize) {}
#else
         SMPProcessor( int id ) : PE( id, &SMP ) {}
#endif

         virtual ~SMPProcessor() {}

         virtual WD & getWorkerWD () const;
         virtual WD & getMasterWD () const;
         virtual BaseThread & createThread ( WorkDescriptor &wd );

         static void prepareConfig ( Config &config );
         // capability query functions
#ifdef SMP_SUPPORTS_ULT
         virtual bool supportsUserLevelThreads () const { return _useUserThreads; }
#else
         virtual bool supportsUserLevelThreads () const { return false; }
#endif

#ifdef SMP_NUMA
         /* Memory space suport */
         virtual void waitInputDependent( uint64_t tag );

         virtual void registerCacheAccessDependent( uint64_t tag, size_t size, bool input, bool output );
         virtual void unregisterCacheAccessDependent( uint64_t tag, size_t size, bool output );
         virtual void registerPrivateAccessDependent( uint64_t tag, size_t size, bool input, bool output );
         virtual void unregisterPrivateAccessDependent( uint64_t tag, size_t size );

         virtual void* getAddressDependent( uint64_t tag );
         virtual void copyToDependent( void *dst, uint64_t tag, size_t size );

         void synchronize( uint64_t tag );
#endif
   };

}
}

#endif
