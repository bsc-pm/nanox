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

#include <iostream>
#include <cstring>
#include "string.h"
#include "sharedmemallocator.hpp"
#include "openclprocessor_decl.hpp"
#include "CL/opencl.h"

using namespace nanos;

SharedMemAllocator::SharedMemAllocator() : _lock() {   
    _init=false;
}

void SharedMemAllocator::initialize(){    
    if (!_init){
        _init=true;
        for (System::ThreadList::iterator it=sys.getWorkersBegin();
            it!=sys.getWorkersEnd() && _allocatingDevice==NULL; it++) {
            BaseThread* bt = it->second;
            if( nanos::ext::OpenCLProcessor *myPE = dynamic_cast<nanos::ext::OpenCLProcessor *>( bt->runningOn() ) ){
                cl_device_type devType;
                myPE->getOpenCLDeviceType(devType);
                if (devType==CL_DEVICE_TYPE_GPU) _allocatingDevice=myPE;
            }
        }        
        //If GPUs are not present, use a CPU
        for (System::ThreadList::iterator it=sys.getWorkersBegin();
            it!=sys.getWorkersEnd() && _allocatingDevice==NULL; it++) {
            BaseThread* bt = it->second;
            if( nanos::ext::OpenCLProcessor *myPE = dynamic_cast<nanos::ext::OpenCLProcessor *>( bt->runningOn() ) ){
                cl_device_type devType;
                myPE->getOpenCLDeviceType(devType);
                if (devType==CL_DEVICE_TYPE_CPU) _allocatingDevice=myPE;
            }
        }
    }
}

void * SharedMemAllocator::allocate( size_t size )
{    
    initialize();
    void* res;
    if (_allocatingDevice==NULL){
        res=malloc(size);
    } else {
        res=_allocatingDevice->allocateSharedMemory(size);    
        _lock.acquire();
        _pinnedChunks[ res ] = size;
        _lock.release();
    }
   
    return res;
}

void SharedMemAllocator::free( void * addr )
{
   initialize();   
  
   if (_allocatingDevice==NULL){
      std::free(addr);
   } else {
     _lock.acquire();
     _pinnedChunks.erase( addr );
     _lock.release();
     _allocatingDevice->freeSharedMemory(addr);
   }
}


bool SharedMemAllocator::isSharedMem( void * address, size_t size )
{
   if (_pinnedChunks.size()==0) return false;
   
   SharedMemMemoryMap::iterator it = _pinnedChunks.lower_bound( address );

   // Perfect match, check size
   if ( it->first == address ) {
      if ( it->second >= size ) return true;

      // Size is bigger than pinned area
      return false;
   }

   // address is lower than any other pinned address
   if ( it == _pinnedChunks.begin() ) return false;

   // It is an intermediate region, check it fits into a pinned area
   it--;

   if ( ( it->first < address ) && ( ( ( size_t ) it->first + it->second ) >= ( ( size_t ) address + size ) ) )
      return true;

   return false;
}

void* SharedMemAllocator::getBasePointer( void * address, size_t size )
{
   //This is likely an error
   if (_pinnedChunks.size()==0) return NULL;
   
   SharedMemMemoryMap::iterator it = _pinnedChunks.lower_bound( address );

   // Perfect match, check size
   if ( it->first == address ) {
      if ( it->second >= size ) return it->first;
   }

   // address is lower than any other pinned address
   if ( it == _pinnedChunks.begin() ) return NULL;

   // It is an intermediate region, check it fits into a pinned area
   it--;

   if ( ( it->first < address ) && ( ( ( size_t ) it->first + it->second ) >= ( ( size_t ) address + size ) ) ){
       return it->first;
   }
   
   return NULL;
}

void SharedMemAllocator::printSharedMemMemoryMap()
{
   std::cout << "PINNED MEMORY CHUNKS" << std::endl;
   for (SharedMemMemoryMap::iterator it = _pinnedChunks.begin(); it != _pinnedChunks.end(); it++ ) {
      std::cout << "|... ";
      std::cout << it->first << " @ " << it->second;
      std::cout << " ...";
   }
   std::cout << "|" << std::endl;
}



