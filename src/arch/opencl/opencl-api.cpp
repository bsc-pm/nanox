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

NANOS_API_DEF(nanos_err_t,nanos_opencl_set_bufferarg, (void* opencl_kernel, int arg_num, const void* pointer)){
   nanos::ext::OpenCLProcessor *pe=( nanos::ext::OpenCLProcessor * ) getMyThreadSafe()->runningOn();
   pe->setKernelBufferArg(opencl_kernel, arg_num, pointer);
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t,nanos_opencl_set_arg, (void* opencl_kernel, int arg_num, size_t size, const void* pointer)){
   nanos::ext::OpenCLProcessor *pe=( nanos::ext::OpenCLProcessor * ) getMyThreadSafe()->runningOn();
   pe->setKernelArg(opencl_kernel, arg_num, size, pointer);

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t,nanos_exec_kernel, (void* opencl_kernel, int work_dim, size_t* ndr_local_size, size_t* ndr_global_size)){
   nanos::ext::OpenCLProcessor *pe=( nanos::ext::OpenCLProcessor * ) getMyThreadSafe()->runningOn();
   pe->execKernel(opencl_kernel, work_dim, ndr_local_size, ndr_global_size);

   return NANOS_OK;
}

NANOS_API_DECL(nanos_err_t,nanos_profile_exec_kernel, (void* opencl_kernel, int work_dim, size_t* ndr_global_size)){
   nanos::ext::OpenCLProcessor *pe=( nanos::ext::OpenCLProcessor * ) getMyThreadSafe()->runningOn();
   pe->profileKernel(opencl_kernel, work_dim, ndr_global_size);

   return NANOS_OK;
}

unsigned int nanos_get_opencl_num_devices (void){
    return nanos::ext::OpenCLConfig::getOpenCLDevicesCount();
}

void nanos_get_opencl_num_devices_( int* numret){
    *numret=nanos::ext::OpenCLConfig::getOpenCLDevicesCount();
}

void * ompss_opencl_malloc ( size_t size )
{
   return nanos::ext::OpenCLProcessor::getSharedMemAllocator().allocate(size);
}

void * nanos_malloc_opencl ( size_t size )
{
	return ompss_opencl_malloc( size );
}

NANOS_API_DEF(void, nanos_opencl_allocate_fortran, ( ptrdiff_t size, void* ptr ))
{
   (*(void**)ptr) = nanos::ext::OpenCLProcessor::getSharedMemAllocator().allocate(size);
}

void ompss_opencl_free ( void * address ) 
{
   nanos::ext::OpenCLProcessor::getSharedMemAllocator().free(address);
} 

void nanos_free_opencl( void * address )
{
	return ompss_opencl_free( address );
}

NANOS_API_DEF(void, nanos_opencl_deallocate_fortran, ( void * address ))
{
   nanos::ext::OpenCLProcessor::getSharedMemAllocator().free(*((void**)address));
   *((void**)address) = 0;
}
