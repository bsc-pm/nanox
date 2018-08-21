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

#ifndef _NANOS_GPU_THREAD
#define _NANOS_GPU_THREAD

#include "gputhread_decl.hpp"
#include "compatibility.hpp"
#include "gpudd.hpp"
#include "gpuevent.hpp"
#include "gpuprocessor.hpp"
#include "pthread.hpp"


namespace nanos {
namespace ext {

int GPUThread::getGPUDevice ()
{
   return _gpuDevice;
}

int GPUThread::getCpuId() const
{
   return _pthread.getCpuId();
}

GenericEvent * GPUThread::createPreRunEvent( WD * wd )
{
   GPUProcessor * pe = ( GPUProcessor * ) this->AsyncThread::runningOn();
#ifdef NANOS_GENERICEVENT_DEBUG
   return NEW GPUEvent( wd, pe->getGPUProcessorInfo()->getInTransferStream(), "Pre-run event" );
#else
   return NEW GPUEvent( wd, pe->getGPUProcessorInfo()->getInTransferStream() );
#endif
}

GenericEvent * GPUThread::createRunEvent( WD * wd )
{
   unsigned int streamIdx = ( wd->getCudaStreamIdx() != -1 ) ? wd->getCudaStreamIdx() : _kernelStreamIdx;
   GPUProcessor * pe = ( GPUProcessor * ) this->AsyncThread::runningOn();

#ifdef NANOS_GENERICEVENT_DEBUG
   return NEW GPUEvent( wd, pe->getGPUProcessorInfo()->getKernelExecStream( streamIdx ), "Run event" );
#else
   return NEW GPUEvent( wd, pe->getGPUProcessorInfo()->getKernelExecStream( streamIdx ) );
#endif
}

GenericEvent * GPUThread::createPostRunEvent( WD * wd )
{
   GPUProcessor * pe = ( GPUProcessor * ) this->AsyncThread::runningOn();
#ifdef NANOS_GENERICEVENT_DEBUG
   return NEW GPUEvent( wd, pe->getGPUProcessorInfo()->getOutTransferStream(), "Post-run event" );
#else
   return NEW GPUEvent( wd, pe->getGPUProcessorInfo()->getOutTransferStream() );
#endif
}

} // namespace ext
} // namespace nanos

#endif
