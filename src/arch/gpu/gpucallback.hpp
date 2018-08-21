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

#ifndef _NANOS_GPU_CALLBACK
#define _NANOS_GPU_CALLBACK

#include "gputhread_decl.hpp"

#include <cuda_runtime.h>


namespace nanos {
namespace ext {

   class GPUCallbackData
   {
      public:

         GPUThread * _thread;
         WD *        _wd;
         size_t      _size;

         // constructors
         GPUCallbackData(  GPUThread * thread ) :  _thread( thread ), _wd( NULL ), _size( 0 ) {}
         GPUCallbackData(  GPUThread * thread, WD * wd ) :  _thread( thread ), _wd( wd ), _size( 0 ) {}
         GPUCallbackData(  GPUThread * thread, size_t size ) :  _thread( thread ), _wd( NULL ), _size( size ) {}

         // destructor
         ~GPUCallbackData() {}


   };

   // Running WD
   void CUDART_CB beforeWDRunCallback( cudaStream_t stream, cudaError_t status, void * data );
   void CUDART_CB afterWDRunCallback( cudaStream_t stream, cudaError_t status, void * data );

   // Copying inputs
   void CUDART_CB beforeAsyncInputCallback( cudaStream_t stream, cudaError_t status, void * data );
   void CUDART_CB afterAsyncInputCallback( cudaStream_t stream, cudaError_t status, void * data );

   // Copying outputs
   void CUDART_CB beforeAsyncOutputCallback( cudaStream_t stream, cudaError_t status, void * data );
   void CUDART_CB afterAsyncOutputCallback( cudaStream_t stream, cudaError_t status, void * data );

   void CUDART_CB registerCUDAThreadCallback( cudaStream_t stream, cudaError_t status, void * data );
   void CUDART_CB unregisterCUDAThreadCallback( cudaStream_t stream, cudaError_t status, void * data );




} // namespace ext
} // namespace nanos

#endif // _NANOS_GPU_CALLBACK
