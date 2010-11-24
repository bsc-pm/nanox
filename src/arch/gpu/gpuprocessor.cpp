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


GPUProcessor::GPUProcessor( int id, int gpuId ) : CachedAccelerator( id, &GPU ),
      _gpuDevice( _deviceSeed++ ), _gpuProcessorTransfers(), _allocator(), _pinnedMemory()
{
   _gpuProcessorInfo = new GPUProcessorInfo( gpuId );
}

void GPUProcessor::init( size_t &memSize )
{
   void * baseAddress = GPUDevice::allocateWholeMemory( memSize );
   _allocator.init( ( uint64_t ) baseAddress, memSize );
   setSize( memSize );

   // Create a list of inputs that have been ordered to transfer but the copy is still not completed
   if ( _gpuProcessorInfo->getInTransferStream() ) {
      delete _gpuProcessorTransfers._pendingCopiesIn;
      _gpuProcessorTransfers._pendingCopiesIn = new GPUMemoryTransferInAsyncList();
   }

   if ( _gpuProcessorInfo->getOutTransferStream() ) {
      // If overlapping outputs is defined, create the list
      delete _gpuProcessorTransfers._pendingCopiesOut;
      _gpuProcessorTransfers._pendingCopiesOut = new GPUMemoryTransferOutAsyncList();
   }
   else {
      // Else, create a 'fake list' which copies outputs synchronously
      delete _gpuProcessorTransfers._pendingCopiesOut;
      _gpuProcessorTransfers._pendingCopiesOut = new GPUMemoryTransferOutSyncList();
   }
}

void GPUProcessor::freeWholeMemory()
{
   GPUDevice::freeWholeMemory( ( void * ) _allocator.getBaseAddress() );
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
   GPUThread &th = *new GPUThread( helper, this, _gpuDevice );

   return th;
}


