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


#include "gpucallback.hpp"
#include "system.hpp"


using namespace nanos;
using namespace ext;


void CUDART_CB nanos::ext::beforeWDRunCallback( cudaStream_t stream, cudaError_t status, void * data )
{
   GPUThread * thread = ( ( GPUCallbackData * ) data )->_thread;
   WD * wd = ( ( GPUCallbackData * ) data )->_wd;

   myThread = ( BaseThread * ) thread->getCUDAThreadInst();

   thread->raiseWDRunEvent( wd );

   delete ( GPUCallbackData * ) data;
}


void CUDART_CB nanos::ext::afterWDRunCallback( cudaStream_t stream, cudaError_t status, void * data )
{
   GPUThread * thread = ( ( GPUCallbackData * ) data )->_thread;
   WD * wd = ( ( GPUCallbackData * ) data )->_wd;

   myThread = ( BaseThread * ) thread->getCUDAThreadInst();

   thread->closeWDRunEvent( wd );

   delete ( GPUCallbackData * ) data;
}


void CUDART_CB nanos::ext::beforeAsyncInputCallback( cudaStream_t stream, cudaError_t status, void * data )
{
   GPUThread * thread = ( ( GPUCallbackData * ) data )->_thread;
   size_t size = ( ( GPUCallbackData * ) data )->_size;

   myThread = ( BaseThread * ) thread->getCUDAThreadInst();

   thread->raiseAsyncInputEvent( size );

   delete ( GPUCallbackData * ) data;
}


void CUDART_CB nanos::ext::afterAsyncInputCallback( cudaStream_t stream, cudaError_t status, void * data )
{
   GPUThread * thread = ( ( GPUCallbackData * ) data )->_thread;
   size_t size = ( ( GPUCallbackData * ) data )->_size;

   myThread = ( BaseThread * ) thread->getCUDAThreadInst();

   thread->closeAsyncInputEvent( size );

   delete ( GPUCallbackData * ) data;
}


void CUDART_CB nanos::ext::beforeAsyncOutputCallback( cudaStream_t stream, cudaError_t status, void * data )
{
   GPUThread * thread = ( ( GPUCallbackData * ) data )->_thread;
   size_t size = ( ( GPUCallbackData * ) data )->_size;

   myThread = ( BaseThread * ) thread->getCUDAThreadInst();

   thread->raiseAsyncOutputEvent( size );

   delete ( GPUCallbackData * ) data;
}


void CUDART_CB nanos::ext::afterAsyncOutputCallback( cudaStream_t stream, cudaError_t status, void * data )
{
   GPUThread * thread = ( ( GPUCallbackData * ) data )->_thread;
   size_t size = ( ( GPUCallbackData * ) data )->_size;

   myThread = ( BaseThread * ) thread->getCUDAThreadInst();

   thread->closeAsyncOutputEvent( size );

   delete ( GPUCallbackData * ) data;
}


void CUDART_CB nanos::ext::registerCUDAThreadCallback( cudaStream_t stream, cudaError_t status, void * data )
{
   GPUThread * thread = ( ( GPUCallbackData * ) data )->_thread;

   NANOS_INSTRUMENT ( sys.getInstrumentation()->incrementMaxThreads(); )

   sys.admitCurrentThread( false );

   thread->setCUDAThreadInst( myThread );
}


void CUDART_CB nanos::ext::unregisterCUDAThreadCallback( cudaStream_t stream, cudaError_t status, void * data )
{
   GPUThread * thread = ( ( GPUCallbackData * ) data )->_thread;

   myThread = ( BaseThread * ) thread->getCUDAThreadInst();

   sys.expelCurrentThread( false );
   myThread->leaveTeam();
}
