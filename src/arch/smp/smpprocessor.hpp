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

#ifndef _NANOS_SMP_PROCESSOR
#define _NANOS_SMP_PROCESSOR

#include "smpthread_fwd.hpp"
#include "smpdevice_decl.hpp"
#include "processingelement_decl.hpp"

#include "config.hpp"

// xlc/icc compilers require the next include to emit the vtable of WDDeque
#include <wddeque.hpp>

//TODO: Make smp independent from pthreads? move it to OS?

namespace nanos {
namespace ext {

   class SMPProcessor : public PE
   {
      private:
         // config variables
         static bool _useUserThreads;
         static size_t _threadsStackSize;
         static size_t _cacheDefaultSize;
         static System::CachePolicyType _cachePolicy;
         unsigned int _bindingId;
         CpuSet _bindingList;
         bool _reserved;
         bool _active;
         unsigned int _futureThreads;

         // disable copy constructor and assignment operator
         SMPProcessor( const SMPProcessor &pe );
         const SMPProcessor & operator= ( const SMPProcessor &pe );


      public:
         // constructors
         SMPProcessor( int bindingId, const CpuSet& bindingList, memory_space_id_t numMemId,
               bool active, unsigned int numaNode, unsigned int socket );

         unsigned int getBindingId() const { return _bindingId; }
         const CpuSet& getBindingList() const { return _bindingList; }

         virtual ~SMPProcessor() {}

         virtual WD & getMultiWorkerWD () const;
         virtual WD & getWorkerWD () const;
         virtual WD & getMasterWD () const;
         virtual BaseThread & createThread ( WorkDescriptor &wd, SMPMultiThread *parent=NULL );
         virtual BaseThread & createMultiThread ( WorkDescriptor &wd, unsigned int numPEs, PE **repPEs );
         SMPThread &associateThisThread(bool untieMaster);

         static void prepareConfig ( Config &config );
         // capability query functions
#ifdef SMP_SUPPORTS_ULT
         virtual bool supportsUserLevelThreads () const { return _useUserThreads; }
#else
         virtual bool supportsUserLevelThreads () const { return false; }
#endif
         bool isReserved() const { return _reserved; }
         void reserve() { _reserved = true; }
         virtual bool isActive() const { return _active; }
         void setActive( bool value = true) { _active = value; }
         //virtual void* getAddressDependent( uint64_t tag );
         //virtual void* waitInputsDependent( WorkDescriptor &work );
         //virtual void* newGetAddressDependent( CopyData const &cd );
         void setNumFutureThreads( unsigned int nthreads );
         unsigned int getNumFutureThreads() const;
   };

} // namespace ext
} // namespace nanos

#endif
