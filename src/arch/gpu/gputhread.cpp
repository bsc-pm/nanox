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

#include "gputhread.hpp"
#include "schedule.hpp"

#include <cuda_runtime.h>

using namespace nanos;
using namespace nanos::ext;


void GPUThread::runDependent ()
{
   WD &work = getThreadWD();
   setCurrentWD( work );
   setNextWD( (WD *) 0 );

   cudaError_t cudaErr = cudaSetDevice( _gpuDevice );
   if (cudaErr != cudaSuccess) warning( "couldn't set the GPU device" );

#if PINNED_CUDA | WC
   cudaErr = cudaSetDeviceFlags( cudaDeviceMapHost );
   if (cudaErr != cudaSuccess) warning( "couldn't set the GPU device flags" );
#endif

   SMPDD &dd = ( SMPDD & ) work.activateDevice( SMP );

   dd.getWorkFct()( work.getData() );
}

void GPUThread::inlineWorkDependent ( WD &wd )
{
   SMPThread::inlineWorkDependent( wd );

#if !NORMAL
   // Get next task in order to prefetch data to device memory
   WD *next = Scheduler::prefetch( ( nanos::BaseThread * ) this, wd );
   setNextWD( next );
   if ( next != 0 ) {
      next->start(false);
   }
#endif

   // Wait for the GPU kernel to finish
   cudaThreadSynchronize();
}
