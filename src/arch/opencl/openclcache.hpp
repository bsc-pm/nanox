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

#ifndef _NANOS_OpenCL_CACHE
#define _NANOS_OpenCL_CACHE

#include "basethread_decl.hpp"
#include "cache_decl.hpp"
#include "copydescriptor_decl.hpp"
#include "simpleallocator.hpp"
#include "system_decl.hpp"
#include "openclutils.hpp"
#include "openclconfig.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <limits>
#include <queue>
#include <map>


#include <cassert>

namespace nanos {
namespace ext {

class OpenCLAdapter;
class OpenCLProcessor;
class OpenCLCache;

class OpenCLCache
{
  friend class OpenCLProcessor;
  
public:
  OpenCLCache(OpenCLAdapter &openclAdapter) : _devCacheSize( 0 ),
                                     _openclAdapter( openclAdapter ) { }

  OpenCLCache( const OpenCLCache &cache ); // Do not implement.
  const OpenCLCache &operator=( const OpenCLCache &cache ); // Do not implement.

public:
   ~OpenCLCache();
   
   void initialize();
   
   void *allocate( size_t size, uint64_t tag);

   void *reallocate( void* addr, size_t size, size_t ceSize );

   void free( void* addr );

   bool copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size );

   bool copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size );
   
   cl_mem getBuffer( void *localSrc, size_t size );
   
   bool copyInBuffer( void *localDst, cl_mem buffer, size_t size );
   
   void *getDeviceBase()
   {
      return reinterpret_cast<void *>( _devAllocator.getBaseAddress() );
   }

   size_t getSize() const { return _devCacheSize; }
   
   cl_mem toMemoryObjSS( void * addr );

private:   
   cl_mem _mainBuffer;    
   size_t _devCacheSize;

   SimpleAllocator _devAllocator;

   OpenCLAdapter &_openclAdapter;
  
   Atomic<unsigned int>    _bytesIn;
   Atomic<unsigned int>    _bytesOut;
   Atomic<unsigned int>    _bytesDevice;
};

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OpenCL_CACHE
