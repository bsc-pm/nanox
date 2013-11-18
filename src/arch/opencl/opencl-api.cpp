
/*************************************************************************************/
/*      Copyright 2013 Barcelona Supercomputing Center                               */
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

#include "nanos-opencl.h"
#include "system.hpp"
#include "openclprocessor.hpp"
#include "opencldd.hpp"
#include <string.h>


using namespace nanos;

NANOS_API_DEF(void *, nanos_opencl_factory, (void *args)) {
    nanos_opencl_args_t *opencl = (nanos_opencl_args_t *) args;
    return (void *) new ext::OpenCLDD(opencl->outline);
}

NANOS_API_DEF(void*, nanos_create_current_kernel, (const char* kernel_name,const char* opencl_code,const char* compiler_opts)){  
    nanos::ext::OpenCLProcessor *pe=( nanos::ext::OpenCLProcessor * ) getMyThreadSafe()->runningOn();
    return pe->createKernel(kernel_name,opencl_code,compiler_opts);
}

NANOS_API_DEF(nanos_err_t,nanos_opencl_set_bufferarg, (void* opencl_kernel, int arg_num, void* pointer)){
   try {
      nanos::ext::OpenCLProcessor *pe=( nanos::ext::OpenCLProcessor * ) getMyThreadSafe()->runningOn();
      pe->setKernelBufferArg(opencl_kernel, arg_num, pointer);
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t,nanos_opencl_set_arg, (void* opencl_kernel, int arg_num, size_t size, void* pointer)){
    try {
      nanos::ext::OpenCLProcessor *pe=( nanos::ext::OpenCLProcessor * ) getMyThreadSafe()->runningOn();
      pe->setKernelArg(opencl_kernel, arg_num, size, pointer);
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t,nanos_exec_kernel, (void* opencl_kernel, int work_dim, size_t* ndr_offset, size_t* ndr_local_size, size_t* ndr_global_size)){
    try {
      nanos::ext::OpenCLProcessor *pe=( nanos::ext::OpenCLProcessor * ) getMyThreadSafe()->runningOn();
      pe->execKernel(opencl_kernel, work_dim, ndr_offset, ndr_local_size, ndr_global_size);
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(unsigned int,nanos_get_opencl_num_devices, (void)){
    return nanos::ext::OpenCLConfig::getOpenCLDevicesCount();
}

NANOS_API_DEF(void *, nanos_malloc_opencl, ( size_t size ))
{
   return nanos::ext::OpenCLProcessor::getSharedMemAllocator().allocate(size);
}

NANOS_API_DEF(intptr_t, nanos_malloc_openclf, ( int size ))
{
   return (intptr_t)nanos::ext::OpenCLProcessor::getSharedMemAllocator().allocate((size_t)size);
}

NANOS_API_DEF( void, nanos_free_opencl, ( void * address ) )
{
   nanos::ext::OpenCLProcessor::getSharedMemAllocator().free(address);
}

NANOS_API_DEF(void, nanos_free_openclf, ( intptr_t address ))
{
   nanos::ext::OpenCLProcessor::getSharedMemAllocator().free((void*)address);
}