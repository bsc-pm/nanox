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
#include "cachedaccelerator.hpp"
#include "copydescriptor_decl.hpp"
#else
#include "processingelement.hpp"
#endif

// xlc/icc compilers require the next include to emit the vtable of WDDeque
#include <wddeque.hpp>

//TODO: Make smp independent from pthreads? move it to OS?

namespace nanos {
namespace ext
{

#ifdef SMP_NUMA

   class SMPProcessor : public nanos::CachedAccelerator<SMPDevice>
#else
   class SMPProcessor : public PE
#endif
   {


      private:
         // config variables
         static bool _useUserThreads;
         static size_t _threadsStackSize;
         static size_t _cacheDefaultSize;
         static System::CachePolicyType _cachePolicy;

         // disable copy constructor and assignment operator
         SMPProcessor( const SMPProcessor &pe );
         const SMPProcessor & operator= ( const SMPProcessor &pe );


      public:
         // constructors
#ifdef SMP_NUMA
         SMPProcessor( int id, int uid ) :
            CachedAccelerator<SMPDevice>( id, &SMP, uid ) {}
#else
         SMPProcessor( int id, int uid ) : PE( id, &SMP, uid ) {}
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
   };

}
}

#endif
