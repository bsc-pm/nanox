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

#ifndef _NANOS_OpenCL_CACHE
#define _NANOS_OpenCL_CACHE

#include "basethread_decl.hpp"
#include "copydescriptor_decl.hpp"
#include "simpleallocator.hpp"
#include "system_decl.hpp"
#include "openclutils.hpp"
#include "openclconfig.hpp"

#ifdef HAVE_OPENCL_OPENCL_H
#include <OpenCL/opencl.h>
#endif

#ifdef HAVE_CL_OPENCL_H
#include <CL/opencl.h>
#endif

#include <limits>
#include <queue>
#include <map>


#include <cassert>

#define ALLOCATOR_START_ADDR 17179869184

namespace nanos {
namespace ext {

class OpenCLAdapter;
class OpenCLProcessor;
class OpenCLCache;

class OpenCLCache
{
  friend class OpenCLProcessor;
  
private:
   cl_mem               _mainBuffer;
   size_t               _devCacheSize;

   SimpleAllocator      _devAllocator;

   OpenCLAdapter       &_openclAdapter;
   OpenCLProcessor     *_processor;      //!< Processor "owner" of this Cache

   Atomic<size_t>       _bytesIn;
   Atomic<size_t>       _bytesOut;
   Atomic<size_t>       _bytesDevice;

public:
  OpenCLCache(OpenCLAdapter &openclAdapter, OpenCLProcessor* processor) : 
             _mainBuffer(), _devCacheSize( 0 ), _devAllocator(), _openclAdapter( openclAdapter ), 
             _processor( processor ), _bytesIn(0), _bytesOut(0), _bytesDevice(0) { }

  OpenCLCache( const OpenCLCache &cache ); // Do not implement.
  const OpenCLCache &operator=( const OpenCLCache &cache ); // Do not implement.

public:
   ~OpenCLCache();
   
   void initialize();
   
   void *allocate( size_t size, uint64_t tag, uint64_t offset);

   void *reallocate( void* addr, size_t size, size_t ceSize );

   void free( void* addr );

   bool copyIn( uint64_t devAddr, uint64_t hostAddr, size_t size, DeviceOps* ops );

   bool copyOut( uint64_t hostAddr, uint64_t devAddr, size_t size, DeviceOps* ops );
   
   cl_mem getBuffer( void *localSrc, size_t size );
   
   bool copyInBuffer( void *localDst, cl_mem buffer, size_t size, DeviceOps *ops );
   
   void *getDeviceBase()
   {
      return reinterpret_cast<void *>( _devAllocator.getBaseAddress() );
   }

   size_t getSize() const { return _devCacheSize; }
   
   cl_mem toMemoryObjSS( const void * addr );
   
   SimpleAllocator& getAllocator()
   {
       return _devAllocator;
   }

   SimpleAllocator const &getConstAllocator() const
   {
       return _devAllocator;
   }

};

} // namespace ext
} // namespace nanos

#endif // _NANOS_OpenCL_CACHE
