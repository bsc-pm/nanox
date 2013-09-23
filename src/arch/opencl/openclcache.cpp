
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

#include "openclcache.hpp"
#include "openclconfig.hpp"
#include "openclprocessor.hpp"

using namespace nanos;
using namespace nanos::ext;

//
// OpenCLCache implementation.
//

OpenCLCache::~OpenCLCache() {
    if (_mainBuffer!=0) clReleaseMemObject(_mainBuffer);
}

void OpenCLCache::initialize() {
    // Compute the amount of cache for the device.
    _devCacheSize = OpenCLConfig::getDevCacheSize();
    //If no device memory specified, allocate 90% of the memory
    if (_devCacheSize==0) _devCacheSize=90;
    //If less than 100 "bytes", specified, assume its a x% of the memory
    if (_devCacheSize <= 100)
        _devCacheSize = _openclAdapter.getGlobalSize()*_devCacheSize/100;

    
    //If device is not a CPU (aka shared memory, allocate the whole memory)
    if (_openclAdapter.getPreallocatesWholeMemory()){
        if (_openclAdapter.allocBuffer(_devCacheSize, _mainBuffer) != CL_SUCCESS)
            fatal0("Not enough memory available on device to allocate requested memory size");
    }

    // Initialize the device allocator.
    _devAllocator.init(0 , _devCacheSize);
}

void *OpenCLCache::deviceAllocate(size_t size) {
    //cl_mem buf;
    

    //if (_openclAdapter.allocBuffer(size, buf) != CL_SUCCESS)
    //    fatal("Device allocation failed");

    void *addr = _devAllocator.allocate(size);
    //Create the buffer
    _openclAdapter.getBuffer(_mainBuffer,(size_t)addr,size);


    //_bufAddrMappings[addr] = buf;

    return addr;
}

void *OpenCLCache::deviceReallocate(void * addr, size_t size, size_t ceSize) {
   
    deviceFree(addr);

    return deviceAllocate(size);
}

void OpenCLCache::deviceFree(void * addr) {
    _devAllocator.free(addr);
    //cl_mem buf = _bufAddrMappings[addr];    
    //cl_int errCode;
    _openclAdapter.freeAddr(addr);
    //if ( errCode != CL_SUCCESS)
    //    warning0("Cannot free device buffer.");
    
    //_bufAddrMappings.erase(_bufAddrMappings.find(addr)); 
}

bool OpenCLCache::deviceCopyIn(void *localDst,
        CopyDescriptor &remoteSrc,
        size_t size) {
    cl_int errCode;   
    
    cl_mem buf = _openclAdapter.getBuffer(_mainBuffer,(size_t)localDst,size);
    
    errCode = _openclAdapter.writeBuffer(buf,
              (void*) remoteSrc.getTag(),
              0,
              size);
    if (errCode != CL_SUCCESS){
        fatal("Buffer writing failed.");
    }
    _bytesIn += ( unsigned int ) size;

    return true;
}

bool OpenCLCache::deviceCopyOut(CopyDescriptor &remoteDst,
        void *localSrc,
        size_t size) {
    cl_int errCode;
    
    cl_mem buf = _openclAdapter.getBuffer(_mainBuffer,(size_t)localSrc,size);
    errCode = _openclAdapter.readBuffer(buf,
                ((void*)remoteDst.getTag()),
                0,
                size);
    
    if (errCode != CL_SUCCESS && localSrc!=0) {        
        fatal("Buffer reading failed.");
    }    
    
    _bytesOut += ( unsigned int ) size;

    return true;
}

bool OpenCLCache::deviceCopyInBuffer(void *localSrc,
        cl_mem remoteBuffer,
        size_t size) {    
    cl_int errCode;
    
    cl_mem buf = _openclAdapter.getBuffer(_mainBuffer,(size_t)localSrc,size);
    
    errCode = _openclAdapter.copyInBuffer(buf,
                remoteBuffer,
                0,
                0,
                size);
    
    if (errCode != CL_SUCCESS) {
        fatal("Buffer copy dev2dev failed.");
    }    
    
    _bytesDevice += ( unsigned int ) size;

    return true;
}


cl_mem OpenCLCache::toMemoryObjSS( void * addr ) { 
    //Creates a buffer from this pointer to the end of the memory (not really correct...)    
    return _openclAdapter.getBuffer(_mainBuffer,(size_t)addr,-1);
}
   

cl_mem OpenCLCache::getBuffer( void *localSrc, size_t size)
{
   return _openclAdapter.getBuffer(_mainBuffer,(size_t)localSrc,size);
}
