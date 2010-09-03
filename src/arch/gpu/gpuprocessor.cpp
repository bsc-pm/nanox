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

#include "gpuprocessor.hpp"
#include "debug.hpp"
#include "schedule.hpp"

#include "cuda_runtime.h"

using namespace nanos;
using namespace nanos::ext;

Atomic<int> GPUProcessor::_deviceSeed = 0;


GPUProcessor::GPUProcessor( int id, int gpuId )
   : Accelerator( id, &GPU ), _gpuDevice( _deviceSeed++ ), _cache(), _pinnedMemory()
{
   _gpuProcessorInfo = new GPUProcessorInfo( gpuId );
}

size_t GPUProcessor::getMaxMemoryAvailable ( int id )
{
   return _gpuProcessorInfo->getMaxMemoryAvailable();
}

WorkDescriptor & GPUProcessor::getWorkerWD () const
{
   SMPDD * dd = new SMPDD( ( SMPDD::work_fct )Scheduler::workerLoop );
   WD *wd = new WD( dd );
   return *wd;
}

WorkDescriptor & GPUProcessor::getMasterWD () const
{
   fatal("Attempting to create a GPU master thread");
}

BaseThread &GPUProcessor::createThread ( WorkDescriptor &helper )
{
   // In fact, the GPUThread will run on the CPU, so make sure it canRunIn( SMP )
   ensure( helper.canRunIn( SMP ), "Incompatible worker thread" );
   GPUThread &th = *new GPUThread( helper,this, _gpuDevice );

   return th;
}

void GPUProcessor::registerCacheAccessDependent( uint64_t tag, size_t size, bool input, bool output )
{
   _cache.registerCacheAccess( tag, size, input, output );
}

void GPUProcessor::unregisterCacheAccessDependent( uint64_t tag, size_t size )
{
   _cache.unregisterCacheAccess( tag, size );
}

void GPUProcessor::flushCacheAccess( uint64_t tag, size_t size )
{
   _cache.flushCacheAccess( tag, size );
}

void GPUProcessor::registerPrivateAccessDependent( uint64_t tag, size_t size, bool input, bool output )
{
   _cache.registerPrivateAccess( tag, size, input, output );
}

void GPUProcessor::unregisterPrivateAccessDependent( uint64_t tag, size_t size )
{
   _cache.unregisterPrivateAccess( tag, size );
}

void* GPUProcessor::getAddressDependent( uint64_t tag )
{
   return _cache.getAddress( tag );
}

void GPUProcessor::copyToDependent( void *dst, uint64_t tag, size_t size )
{
   _cache.copyTo( dst, tag, size );
}


