
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

#ifndef _OpenCL_DEVICE
#define _OpenCL_DEVICE

#include "opencldevice_decl.hpp"
#include "openclprocessor.hpp" 
#include "deviceops.hpp"

using namespace nanos;
using namespace nanos::ext;

OpenCLDevice::OpenCLDevice( const char *name ) : Device( name ) { }

void *OpenCLDevice::memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, uint64_t targetHostAddr) const
{ 
   nanos::ProcessingElement * pe = &(mem.getPE());
   if( OpenCLProcessor *proc = dynamic_cast<OpenCLProcessor *>( pe ) )
      return proc->allocate( size , targetHostAddr);


   fatal( "Can allocate only on OpenCLProcessor" );
}

//void *OpenCLDevice::realloc( void * address,
//                          size_t size,
//                          size_t ceSize,
//                          ProcessingElement *pe )
//{
//   if( OpenCLProcessor *proc = dynamic_cast<OpenCLProcessor *>( pe ) )
//      return proc->realloc( address, size, ceSize );
//   fatal( "Can reallocate only on OpenCLProcessor" );
//}

void OpenCLDevice::memFree( uint64_t addr, SeparateMemoryAddressSpace &mem ) const
{
    nanos::ProcessingElement * pe = &(mem.getPE());

    if( OpenCLProcessor *proc = dynamic_cast<OpenCLProcessor *>( pe ) )
      return proc->free( (void*) addr );


   fatal( "Can free only on OpenCLProcessor" );
}

void OpenCLDevice::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const
{
   nanos::ProcessingElement * pe = &(mem.getPE());
   if( OpenCLProcessor *proc = dynamic_cast<OpenCLProcessor *>( pe ) )
   {
      // Current thread is not the device owner: instead of doing the copy, add
      // it to the pending transfer list.
//      if( myThread->runningOn() != pe )
//         return proc->asyncCopyIn( localDst, remoteSrc, size );
//
//      // We can do a synchronous copy.
//      else
        proc->copyIn( devAddr, hostAddr, len, ops );
   }
}

void OpenCLDevice::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const
{
   nanos::ProcessingElement * pe = &(mem.getPE());
   if( OpenCLProcessor *proc = dynamic_cast<OpenCLProcessor *>( pe ) )
   {
      // Current thread is not the device owner: instead of doing the copy, add
      // it to the pending transfer list.
//      if( myThread->runningOn() != pe )
//         return proc->asyncCopyOut( remoteDst, localSrc, size );
//
//      // We can do a synchronous copy.
//      else
        proc->copyOut( hostAddr, devAddr, len, ops );
   }
}

bool OpenCLDevice::_copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const
{
    //If user disabled devToDev copies (sometimes they give bad performance...)
    if (nanos::ext::OpenCLConfig::getDisableDev2Dev()) return false;
    nanos::ext::OpenCLProcessor *procDst = (nanos::ext::OpenCLProcessor *)( &(memDest.getPE()) );
    nanos::ext::OpenCLProcessor *procSrc = (nanos::ext::OpenCLProcessor *)( &(memOrig.getPE()) );
    //If both devices are in the same vendor/context do a real copy in       
   //If shared memory, no need to copy (I hope, all OCL devices should share the same memory space...)
    if (procDst->getContext()==procSrc->getContext() && !OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) devOrigAddr, len)) {       
        ops->addOp();
        cl_mem buf=procSrc->getBuffer( (void*) devOrigAddr, len);
        procDst->copyInBuffer( (void*) devDestAddr, buf, len);
        ops->completeOp(); 
        if ( f ) {
           (*f)(); 
        }
        return true;
    }
    return false;
}


void OpenCLDevice::_getFreeMemoryChunksList( SeparateMemoryAddressSpace &mem, SimpleAllocator::ChunkList &list ) const {
    nanos::ext::OpenCLProcessor *pe = (nanos::ext::OpenCLProcessor *)&(mem.getPE());
    pe->getCacheAllocator().getFreeChunksList(list);
}

std::size_t OpenCLDevice::getMemCapacity( SeparateMemoryAddressSpace &mem ) const {
    nanos::ext::OpenCLProcessor *pe = (nanos::ext::OpenCLProcessor *)&(mem.getPE());
    return pe->getCacheAllocator().getCapacity();
}

#endif // _OpenCL_DEVICE
