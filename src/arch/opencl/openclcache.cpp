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

#include "openclcache.hpp"
#include "openclconfig.hpp"
#include "openclprocessor.hpp"
#include "deviceops.hpp"
#include "openclthread_decl.hpp"
#include "openclevent.hpp"

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
        if (_openclAdapter.allocBuffer(_devCacheSize,NULL, _mainBuffer) != CL_SUCCESS)
            fatal("Not enough memory available on device to allocate requested memory size");
    } else {
        _mainBuffer=NULL;
    }

    // Initialize the device allocator.
    _devAllocator.init(ALLOCATOR_START_ADDR, _devCacheSize);
}

void *OpenCLCache::allocate(size_t size, uint64_t tag, uint64_t offset) {
    //Shared memory buffers are already allocated
    //We only need to search for them with tag +1 (they are internally stored in different address)
    if (OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) tag, size)){
        _openclAdapter.getBuffer(_devAllocator,_mainBuffer,tag,size);
        //if (buf==NULL){
        //    return NULL;
        //}
        return (void*)(tag);
    } else {
        //Create the buffer
        size=size+offset;
        _devAllocator.lock();
        void* addr=(void*) _devAllocator.allocate(size);
        _devAllocator.unlock();
        if (addr==NULL) return NULL;
        cl_mem buf=_openclAdapter.createBuffer(_mainBuffer,reinterpret_cast<uint64_t>(addr),size,(void*)tag);
        if (buf==NULL){   
            _devAllocator.lock();
            _devAllocator.free(addr);
            _devAllocator.unlock();
            return NULL;
        }
        
        return (void*)((uint64_t)addr+offset);
    }
}

void *OpenCLCache::reallocate(void * addr, size_t size, size_t ceSize) {
   
    free(addr);

    return allocate(size, (uint64_t) addr, 0);
}

void OpenCLCache::free(void * addr) {
    if (OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) addr, 1)) return;
    _devAllocator.free(addr);
    _openclAdapter.freeAddr(addr);
}

bool OpenCLCache::copyIn(uint64_t devAddr,
        uint64_t hostAddr,
        size_t size, DeviceOps *ops) {
    //If shared memory, no need to copy
    cl_int errCode;
    cl_mem buf = _openclAdapter.getBuffer(_devAllocator,_mainBuffer, devAddr,size);

    ops->addOp();
    // Copy from host memory to device memory
    nanos::ext::OpenCLThread * thread = ( nanos::ext::OpenCLThread * ) _processor->getOpenCLThread();
    OpenCLEvent * evt = (OpenCLEvent*) thread->createPreRunEvent( thread->getCurrentWD() );
#ifdef NANOS_GENERICEVENT_DEBUG
    evt->setDescription( evt->getDescription() + " copy input: " + toString<uint64_t>( remoteSrc.getTag() ) );
#endif
    evt->setCreated();
#ifdef NANOS_GENERICEVENT_DEBUG
    evt->setDescription( evt->getDescription() + " action:DeviceOps::completeOp" );
#endif
    Action * action = new_action( ( ActionMemFunPtr0<DeviceOps>::MemFunPtr0 ) &DeviceOps::completeOp, ops );
    evt->addNextAction( action );
   
    errCode = _openclAdapter.writeBuffer(buf,
              (void*) hostAddr,
              0,
              size,
              &_bytesIn, 
              evt->getCLEvent() );
    
    evt->setPending();

    thread->addEvent( evt );
    
    if (errCode != CL_SUCCESS){
        fatal("Buffer writing failed. Check if you are filling GPU's memory with error" << errCode);
    }
    return true;
}


bool OpenCLCache::copyOut(uint64_t hostAddr,
        uint64_t devAddr,
        size_t size,
        DeviceOps *ops) {       
    //If shared memory, no need to copy
    cl_int errCode;

    cl_mem buf = _openclAdapter.getBuffer(_devAllocator,_mainBuffer, devAddr,size);
    
    ops->addOp();
    // Copy from host memory to device memory
    nanos::ext::OpenCLThread * thread = ( nanos::ext::OpenCLThread * ) _processor->getOpenCLThread();
    OpenCLEvent * evt = (OpenCLEvent*) thread->createPreRunEvent( thread->getCurrentWD() );
#ifdef NANOS_GENERICEVENT_DEBUG
    evt->setDescription( evt->getDescription() + " copy input: " + toString<uint64_t>( remoteSrc.getTag() ) );
#endif
    evt->setCreated();
#ifdef NANOS_GENERICEVENT_DEBUG
    evt->setDescription( evt->getDescription() + " action:DeviceOps::completeOp" );
#endif
    Action * action = new_action( ( ActionMemFunPtr0<DeviceOps>::MemFunPtr0 ) &DeviceOps::completeOp, ops );
    evt->addNextAction( action );
    
    errCode = _openclAdapter.readBuffer(buf,
                ((void*)hostAddr),
                0,
                size,
                &_bytesOut, 
                evt->getCLEvent() );
    
    evt->setPending();

    thread->addEvent( evt );

    if (errCode != CL_SUCCESS && devAddr!=0) {        
        fatal("Buffer reading failed with error" << errCode);
    }    
    return true;
}

bool OpenCLCache::copyInBuffer(void *localSrc,
        cl_mem remoteBuffer,
        size_t size,
        DeviceOps *ops) {            
    cl_int errCode;

    cl_mem buf = _openclAdapter.getBuffer(_devAllocator,_mainBuffer, reinterpret_cast<uint64_t>(localSrc),size);
        
    ops->addOp();
    // Copy from host memory to device memory
    nanos::ext::OpenCLThread * thread = ( nanos::ext::OpenCLThread * ) _processor->getOpenCLThread();
    OpenCLEvent * evt = (OpenCLEvent*) thread->createPreRunEvent( thread->getCurrentWD() );
    
#ifdef NANOS_GENERICEVENT_DEBUG
    evt->setDescription( evt->getDescription() + " copy input: " + toString<uint64_t>( remoteSrc.getTag() ) );
#endif
    evt->setCreated();
#ifdef NANOS_GENERICEVENT_DEBUG
    evt->setDescription( evt->getDescription() + " action:DeviceOps::completeOp" );
#endif
    Action * action = new_action( ( ActionMemFunPtr0<DeviceOps>::MemFunPtr0 ) &DeviceOps::completeOp, ops );
    evt->addNextAction( action );
    errCode = _openclAdapter.copyInBuffer(buf,
                remoteBuffer,
                0,
                0,
                size,
                evt->getCLEvent() );
    
    evt->setPending();

    thread->addEvent( evt );
    
    if (errCode != CL_SUCCESS) {
        fatal("Buffer copy dev2dev failed.");
    }    
    
    _bytesDevice += size;

    return true;
}


cl_mem OpenCLCache::toMemoryObjSS( const void * addr ) { 
    void* addr_aux=const_cast<void*>(addr);
    cl_mem buf= _openclAdapter.getBuffer(_devAllocator,_mainBuffer, reinterpret_cast<uint64_t>(addr),_openclAdapter.getSizeFromCache(reinterpret_cast<uint64_t>(addr)));
    if ( _openclAdapter.getUseHostPtr() || OpenCLProcessor::getSharedMemAllocator().isSharedMem( addr_aux, 1)){
        cl_event ev;
       _openclAdapter.unmapBuffer(buf,addr_aux,0,1,ev);
       cl_int err = 0;
       if (ev != NULL)
           err=clWaitForEvents(1,&ev);
       if (err != CL_SUCCESS) {
          fatal("Error while unmmaping buffer");
       }   
    }
    return buf;
}
   

cl_mem OpenCLCache::getBuffer( void *localSrc, size_t size)
{
   return _openclAdapter.getBuffer(_devAllocator,_mainBuffer,reinterpret_cast<uint64_t>(localSrc),size);
}
