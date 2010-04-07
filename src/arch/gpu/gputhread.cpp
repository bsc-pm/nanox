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
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include <iostream>
#include <sched.h>


// CUDA
#include <cuda_runtime.h>

using namespace nanos;
using namespace nanos::ext;


Atomic<int> GPUThread::_deviceSeed = 0;

/*
void * gpu_bootthread ( void *arg )
{
   GPUThread *self = static_cast<GPUThread *>( arg );

   self->run();

   pthread_exit ( 0 );
}
*/

/*
void GPUThread::start ()
{
   std::cout << "GPUThread::start()" << std::endl;

   pthread_attr_t attr;
   pthread_attr_init(&attr);

   // user-defined stack size
   if ( _stackSize > 0 ) {
     // TODO: check alignment?
     if ( _stackSize < PTHREAD_STACK_MIN ) {
       warning("specified thread stack too small, adjusting it to minimum size");
       _stackSize = PTHREAD_STACK_MIN;
     }

     char *stack = new char[_stackSize];

     if ( stack == NULL || pthread_attr_setstack( &attr, stack, _stackSize ) )
       warning("couldn't create pthread stack");
   }


   if ( pthread_create( &_pth, &attr, gpu_bootthread, this ) )
      fatal( "couldn't create thread" );
}
*/

void GPUThread::runDependent ()
{
   std::cout << "GPUThread::runDependent()" << std::endl;
   WD &work = getThreadWD();
   setCurrentWD( work );

   cudaError_t cudaErr = cudaSetDevice( _gpuDevice );
   if (cudaErr != cudaSuccess) warning( "couldn't set the GPU device" );

   GPUDD &dd = ( GPUDD & ) work.activateDevice( GPU );

   dd.getWorkFct()( work.getData() );
}
/*
void GPUThread::join ()
{
   std::cout << "GPUThread::join()" << std::endl;
   pthread_join( _pth,NULL );
   joined();
}
*/
/*
void GPUThread::bind( void )
{
   std::cout << "GPUThread::bind()" << std::endl;
   cpu_set_t cpu_set;
   int cpu_id = getCpuId();

   ensure( ( ( cpu_id >= 0 ) && ( cpu_id < CPU_SETSIZE ) ), "invalid value for cpu id" );
   CPU_ZERO( &cpu_set );
   CPU_SET( cpu_id, &cpu_set );
   verbose( "Binding thread " << getId() << " to cpu " << cpu_id );
   sched_setaffinity( ( pid_t ) 0, sizeof( cpu_set ), &cpu_set );
}
*/
/*
void GPUThread::yield()
{
   std::cout << "GPUThread::yield()" << std::endl;
   if (sched_yield() != 0)
      warning("sched_yield call returned an error");
}
*/
/*
// This is executed in between switching stacks
void GPUThread::switchHelperDependent ( WD *oldWD, WD *newWD, void *oldState  )
{
   std::cout << "GPUThread::switchHelperDependent()" << std::endl;
   GPUDD & dd = ( GPUDD & )oldWD->getActiveDevice();
   dd.setState( (intptr_t *) oldState );
}
*/
/*
void GPUThread::inlineWorkDependent ( WD &wd )
{
   std::cout << "GPUThread::inlineWorkDependent()" << std::endl;
   GPUDD &dd = ( GPUDD & )wd.getActiveDevice();
   ( dd.getWorkFct() )( wd.getData() );
}
*/

#if 0
void GPUThread::switchTo ( WD *wd, SchedulerHelper *helper )
{
   std::cout << "GPUThread::switchTo()" << std::endl;
   // wd MUST have an active GPU Device when it gets here
   ensure( wd->hasActiveDevice(),"WD has no active GPU device" );
   GPUDD &dd = ( GPUDD & )wd->getActiveDevice();
   ensure( dd.hasStack(), "DD has no stack for ULT");
/*
   ::switchStacks(
       ( void * ) getCurrentWD(),
       ( void * ) wd,
       ( void * ) dd.getState(),
       ( void * ) helper );
*/
}

void GPUThread::exitTo ( WD *wd, SchedulerHelper *helper)
{
   std::cout << "GPUThread::exitTo()" << std::endl;
   // wd MUST have an active GPU Device when it gets here
   ensure( wd->hasActiveDevice(),"WD has no active GPU device" );
   GPUDD &dd = ( GPUDD & )wd->getActiveDevice();
   ensure( dd.hasStack(), "DD has no stack for ULT");

   //TODO: optimize... we don't really need to save a context in this case
/*
   ::switchStacks(
      ( void * ) getCurrentWD(),
      ( void * ) wd,
      ( void * ) dd.getState(),
      ( void * ) helper );
*/
}
#endif
