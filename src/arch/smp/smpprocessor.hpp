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
         SMPProcessor( int id );

         virtual ~SMPProcessor() {}

         virtual WD & getMultiWorkerWD () const;
         virtual WD & getWorkerWD () const;
         virtual WD & getMasterWD () const;
         virtual BaseThread & createThread ( WorkDescriptor &wd, SMPMultiThread *parent=NULL );
         virtual BaseThread & createMultiThread ( WorkDescriptor &wd, unsigned int numPEs, PE **repPEs );

         static void prepareConfig ( Config &config );
         // capability query functions
#ifdef SMP_SUPPORTS_ULT
         virtual bool supportsUserLevelThreads () const { return _useUserThreads; }
#else
         virtual bool supportsUserLevelThreads () const { return false; }
#endif
         virtual bool isGPU () const { return false; }
         //virtual void* getAddressDependent( uint64_t tag );
         //virtual void* waitInputsDependent( WorkDescriptor &work );
         virtual void* newGetAddressDependent( CopyData const &cd );
         virtual bool supportsDirectTransfersWith(ProcessingElement const & pe) const;
   };

}
}

#endif
