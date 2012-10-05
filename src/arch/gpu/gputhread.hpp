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

#ifndef _NANOS_GPU_THREAD
#define _NANOS_GPU_THREAD

#include "gputhread_decl.hpp"
#include "compatibility.hpp"
#include "gpudd.hpp"
#include "gpuprocessor.hpp"


namespace nanos {
namespace ext
{

int GPUThread::getGPUDevice ()
{
   return _gpuDevice;
}

void GPUThread::enableWDClosingEvents ()
{
   _wdClosingEvents = true;
}

void * GPUThread::getCUBLASHandle()
{
   ensure( _cublasHandle, "Trying to use CUBLAS handle without initializing CUBLAS library (please, use NX_GPUCUBLASINIT=yes)" );
   return _cublasHandle;
}

inline int GPUThread::adjustBind( int cpu_id )
{
   int gpuCount = nanos::ext::GPUConfig::getGPUCount();
   int gpusPerSocket = gpuCount / sys.getNumSockets();
   
   // Sanity check
   if ( gpuCount % sys.getNumSockets() != 0 ){
      warning( gpuCount << " GPUs, " << sys.getNumSockets() << " NUMA nodes." );
      fatal( "The number of GPUs is not multiple of the number of NUMA nodes." );
   }
      
   if ( gpuCount > 0 )
   {
      if ( sys.getBindingStride() == 1 && sys.getBindingStart() == 0 )
      {
         // Try to guess the gpu number
         // TODO (gmiranda): I assume the device id matches the gpu thread number
         int gpuIndex = _gpuDevice;
         fprintf( stderr, "GPU thread %d goes to %d\n", cpu_id, sys.getCoresPerSocket()*( 1 + gpuIndex / gpusPerSocket)- ( 1 + ( gpuCount - gpuIndex - 1 ) % gpusPerSocket ));
         //cpu_id = sys.getCoresPerSocket()*(1 + gpuIndex ) - 1;
         cpu_id = sys.getNumSockets()*( 1 + gpuIndex / gpusPerSocket)
            - ( 1 + ( gpuCount - gpuIndex - 1 ) % gpusPerSocket );
      }
      else 
      {
         warning( "Cannot split GPU threads over the available sockets "
            << "when using non-default binding stride and/or offset." );
      }
   }
   
   return cpu_id;
}


}
}

#endif
