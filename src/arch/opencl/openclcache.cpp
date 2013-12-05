
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
#include "deviceops.hpp"

using namespace nanos;
using namespace nanos::ext;

//
// OpenCLCache implementation.
//

OpenCLCache::~OpenCLCache() {
    if (_mainBuffer!=NULL) clReleaseMemObject(_mainBuffer);
}

void OpenCLCache::initialize() {
    // Compute the amount of cache for the device.
    _devCacheSize = OpenCLConfig::getDevCacheSize();
    //If no device memory specified, allocate 90% of the memory
    if (_devCacheSize==0) _devCacheSize=90;
    //If less than 100 "bytes", specified, assume its a x% of the memory
    if (_devCacheSize <= 100)
        _devCacheSize = (size_t)((double)_openclAdapter.getGlobalSize()*_devCacheSize/100);

    
    //If device is not a CPU (aka shared memory, allocate the whole memory)
    if (_openclAdapter.getPreallocatesWholeMemory()){
        if (_openclAdapter.allocBuffer(_devCacheSize, _mainBuffer) != CL_SUCCESS)
            fatal0("Not enough memory available on device to allocate requested memory size");
    } else {
        _mainBuffer=NULL;
    }

    // Initialize the device allocator.
    _devAllocator.init(4 , _devCacheSize);
}

void *OpenCLCache::allocate(size_t size, uint64_t tag) {
    //Shared memory buffers are already allocated
    //We only need to search for them with tag +1 (they are internally stored in different address)
    if (OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) tag, size)){
        cl_mem buf=_openclAdapter.getBuffer(_devAllocator,_mainBuffer,(size_t)tag+1,size);
        if (buf==NULL){
            return NULL;
        }
        return (void*)(tag+1);
    } else {
        _devAllocator.lock();
        void *addr = _devAllocator.allocate(size);
        _devAllocator.unlock();
        if (addr==NULL) return NULL;
        //Create the buffer
        cl_mem buf=_openclAdapter.createBuffer(_mainBuffer,(size_t)addr,size);
        if (buf==NULL){
            return NULL;
        }

        return addr;
    }
}

void *OpenCLCache::reallocate(void * addr, size_t size, size_t ceSize) {
   
    free(addr);

    return allocate(size, (uint64_t) addr);
}

void OpenCLCache::free(void * addr) {
    _openclAdapter.freeAddr(addr);
    _devAllocator.free(addr);
}

bool OpenCLCache::copyIn(uint64_t devAddr,
        uint64_t hostAddr,
        size_t size, DeviceOps *ops) {
    //If shared memory, no need to copy
    cl_int errCode;
    if (OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) hostAddr, size))
    {
        cl_mem buf=_openclAdapter.getBuffer(_devAllocator,_mainBuffer,(size_t)hostAddr,size);  
        errCode = _openclAdapter.unmapBuffer(buf,
              (void*) hostAddr,
              0,
              size);
       // ops->completeOp();
        if (errCode != CL_SUCCESS){
            fatal("Buffer unmap failed.");
        }
    } else {
        cl_mem buf = _openclAdapter.getBuffer(_devAllocator,_mainBuffer,(size_t)devAddr,size);    

        errCode = _openclAdapter.writeBuffer(buf,
                  (void*) hostAddr,
                  0,
                  size);
       // ops->completeOp();
        if (errCode != CL_SUCCESS){
            fatal("Buffer writing failed.");
        }
        _bytesIn += ( unsigned int ) size;
    }
    return true;
}


bool OpenCLCache::copyOut(uint64_t hostAddr,
        uint64_t devAddr,
        size_t size,
        DeviceOps *ops) {
    //If shared memory, no need to copy
    if (OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) hostAddr, size)){
        cl_int errCode;
        
        cl_mem buf = _openclAdapter.getBuffer(_devAllocator,_mainBuffer,(size_t)devAddr,size);
        errCode = _openclAdapter.mapBuffer(buf,
                    ((void*)hostAddr),
                    0,
                    size);

        //ops->completeOp();
        if (errCode != CL_SUCCESS && devAddr!=0) {        
            fatal("Buffer mapping failed.");
        }            
    } else {
        cl_int errCode;

        cl_mem buf = _openclAdapter.getBuffer(_devAllocator,_mainBuffer,(size_t)devAddr,size);
        errCode = _openclAdapter.readBuffer(buf,
                    ((void*)hostAddr),
                    0,
                    size);
       // ops->completeOp();

        if (errCode != CL_SUCCESS && devAddr!=0) {        
            fatal("Buffer reading failed.");
        }    

        _bytesOut += ( unsigned int ) size;
    }
    return true;
}

bool OpenCLCache::copyInBuffer(void *localSrc,
        cl_mem remoteBuffer,
        size_t size) {            
    cl_int errCode;
    
    cl_mem buf = _openclAdapter.getBuffer(_devAllocator,_mainBuffer,(size_t)localSrc,size);
    
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
    return _openclAdapter.getBuffer(_devAllocator,_mainBuffer,(size_t)addr,0, /* unkownsize*/ true);
}
   

cl_mem OpenCLCache::getBuffer( void *localSrc, size_t size)
{
   return _openclAdapter.getBuffer(_devAllocator,_mainBuffer,(size_t)localSrc,size);
}
