
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
    //Shared memory buffers were already allocated
    if (OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) tag, size)){
        cl_mem buf=_openclAdapter.getBuffer(_mainBuffer,(size_t)tag,size);
        if (buf==NULL){
            return CACHE_ALLOC_ERROR;
        }
        return (void*)tag;
    }
    //cl_mem buf;
    

    //if (_openclAdapter.allocBuffer(size, buf) != CL_SUCCESS)
    //    fatal("Device allocation failed");

    void *addr = _devAllocator.allocate(size);
    if (addr==NULL) return CACHE_ALLOC_ERROR;
    //Create the buffer
    cl_mem buf=_openclAdapter.getBuffer(_mainBuffer,(size_t)addr,size);
    if (buf==NULL){
        return CACHE_ALLOC_ERROR;
    }

    //_bufAddrMappings[addr] = buf;

    return addr;
}

void *OpenCLCache::reallocate(void * addr, size_t size, size_t ceSize) {
   
    free(addr);

    return allocate(size, (uint64_t) addr);
}

void OpenCLCache::free(void * addr) {
    //User must free shared memory buffers manually
    //if (OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) addr, 1)) return;
    _devAllocator.free(addr);
    //cl_mem buf = _bufAddrMappings[addr];    
    //cl_int errCode;
    _openclAdapter.freeAddr(addr);
    //if ( errCode != CL_SUCCESS)
    //    warning0("Cannot free device buffer.");
    
    //_bufAddrMappings.erase(_bufAddrMappings.find(addr)); 
}

bool OpenCLCache::copyIn(void *localDst,
        CopyDescriptor &remoteSrc,
        size_t size) {
    //If shared memory, no need to copy
    cl_int errCode;
    if (OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) remoteSrc.getTag(), size))
    {
        cl_mem buf=_openclAdapter.getBuffer(_mainBuffer,(size_t)remoteSrc.getTag(),size);  
        errCode = _openclAdapter.unmapBuffer(buf,
              (void*) remoteSrc.getTag(),
              0,
              size);
        if (errCode != CL_SUCCESS){
            fatal("Buffer unmap failed.");
        }
        return true;
    }
    
    
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


bool OpenCLCache::copyOut(CopyDescriptor &remoteDst,
        void *localSrc,
        size_t size) {
    //If shared memory, no need to copy
    if (OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) remoteDst.getTag(), size)){
        cl_int errCode;
        
        cl_mem buf = _openclAdapter.getBuffer(_mainBuffer,(size_t)localSrc,size);
        errCode = _openclAdapter.mapBuffer(buf,
                    ((void*)remoteDst.getTag()),
                    0,
                    size);

        if (errCode != CL_SUCCESS && localSrc!=0) {        
            fatal("Buffer mapping failed.");
        }            
        return true;
    }
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

bool OpenCLCache::copyInBuffer(void *localSrc,
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
