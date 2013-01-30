/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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
#include <CL/opencl.h>

#include <limits>
#include <queue>
#include <map>


#include <cassert>

namespace nanos {
namespace ext {

class OpenCLAdapter;
class OpenCLProcessor;
class OpenCLCache;

class Address
{
public:
   Address( void *addr,
            OpenCLCache &cache ) : _addr( addr ),
                                _cache( cache ) { }
public:
   void *getAllocAddr() const { return _addr; }

   void *getRaw() const { return _addr; }

public:
   inline bool isHostOperation() const;

private:
   void *_addr;
   OpenCLCache &_cache;
};

class OpenCLMemDump
{
protected:
   static Lock _dumpLock;

protected:
   void dumpCopyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size );
   void dumpCopyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size );

};

class OpenCLCache : public OpenCLMemDump
{
    friend class OpenCLProcessor;
public:
  OpenCLCache(OpenCLAdapter &openclAdapter) : _devCacheSize( 0 ),
                                     _openclAdapter( openclAdapter ) { }

  OpenCLCache( const OpenCLCache &cache ); // Do not implement.
  const OpenCLCache &operator=( const OpenCLCache &cache ); // Do not implement.

public:
   void initialize();
   
   void *allocate( size_t size )
   {
         return deviceAllocate( size );
   }

   void *deviceAllocate( size_t size );

   void *reallocate( void * addr, size_t size, size_t ceSize )
   {
      return deviceReallocate( addr, size, ceSize );

      fatal( "Cannot realloc in different address space" );
   }

   void *deviceReallocate( void* addr, size_t size, size_t ceSize );

   void free( void* addr )
   {
         deviceFree( addr );
   }

   void deviceFree( void* addr );

   bool copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size )
   {
         return deviceCopyIn( localDst, remoteSrc, size );
   }

   bool deviceCopyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size );

   bool copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size )
   {
         return deviceCopyOut( remoteDst, localSrc, size );
   }

   bool deviceCopyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size );

public:

   void *getDeviceBase()
   {
      return reinterpret_cast<void *>( _devAllocator.getBaseAddress() );
   }

   size_t getSize() const { return _devCacheSize; }
   
   cl_mem toMemoryObjSS( void * id ) { return _bufAddrMappings[id]; }
   

private:
   size_t _devCacheSize;

   SimpleAllocator _devAllocator;

   OpenCLAdapter &_openclAdapter;
   
   std::map<void *, cl_mem> _devBufAddrMappings;
   std::map<void *, cl_mem> _bufAddrMappings;
   
   Atomic<unsigned int>    _bytesIn;
   Atomic<unsigned int>    _bytesOut;
   Atomic<unsigned int>    _bytesDevice;
};

template <typename DstTy, typename SrcTy>
class DMATransfer
{
public:
   static const uint64_t MaxId;

private:
   static Atomic<uint64_t> _freeId;

public:
   DMATransfer( DstTy &dst,
                SrcTy &src,
                size_t size ) : _id( _freeId.fetchAndAdd() ),
                                _dst( dst ),
                                _src( src ),
                                _size( size ) {
      if( _id > MaxId )
         fatal( "No more DMA transfer IDs" );
   }

public:
   DstTy &getDestination() const { return const_cast<DstTy &>( _dst ); }
   SrcTy &getSource() const { return const_cast<SrcTy &>( _src ); }
   size_t getSize() const { return _size; }

public:
   void prioritize() { _id = MaxId; }

public:
   bool operator>( const DMATransfer &trans ) const { return _id > trans._id; }

   bool relatedTo( uint64_t address );

public:
   void dump() const;

private:
   uint64_t _id;

   DstTy _dst;
   SrcTy _src;
   size_t _size;
};

template <> inline
bool DMATransfer<void *, CopyDescriptor>::relatedTo( uint64_t address )
{
   return _src.getTag() == address;
}

template <> inline
bool DMATransfer<CopyDescriptor, void *>::relatedTo( uint64_t address )
{
   return _dst.getTag() == address;
}

typedef DMATransfer<void *, CopyDescriptor> DMAInTransfer;
typedef DMATransfer<CopyDescriptor, void *> DMAOutTransfer;

// The standard priority queue uses std::less as comparator, but since transfer
// are ordered according to a logical clock starting from 0 and we want to
// prioritize old transfers, we need to used the greater comparator.
template <typename TransferTy>
class DMATransferQueue : public std::priority_queue<TransferTy,
                                                    std::vector<TransferTy>,
                                                    std::greater<TransferTy>
                                                   >
{
public:
   typedef typename std::vector<TransferTy>::iterator iterator;

public:
   void prioritize( uint64_t address )
   {
      // Augment priority to all related transfers.
      for(iterator i = c.begin(), e = c.end(); i != e; ++i)
         if( i->relatedTo( address ) )
            i->prioritize();

      // Rebuilt the heap.
      std::make_heap(c.begin(), c.end(), comp);
   }

protected:
   using std::priority_queue<TransferTy,
                             std::vector<TransferTy>,
                             std::greater<TransferTy>
                            >::c;
   using std::priority_queue<TransferTy,
                             std::vector<TransferTy>,
                             std::greater<TransferTy>
                            >::comp;
};

typedef DMATransferQueue<DMAInTransfer> DMAInTransferQueue;
typedef DMATransferQueue<DMAOutTransfer> DMAOutTransferQueue;

class OpenCLDMA : public OpenCLMemDump
{
public:
   OpenCLDMA( OpenCLProcessor &proc ) : _proc( proc ) { }

   OpenCLDMA( const OpenCLDMA &dma ); // Do not implement.
   OpenCLDMA &operator=(const OpenCLDMA &dma); // Do not implement.

public:
   bool copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size );
   bool copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size );

   void syncTransfer( uint64_t hostAddress );
   void execTransfers();

private:
   Lock _lock;

   DMAInTransferQueue _ins;
   DMAOutTransferQueue _outs;

   OpenCLProcessor &_proc;
};


std::ostream &operator<<( std::ostream &os, size_t size );

std::ostream &operator<<( std::ostream &os,
                          const DMATransfer<void *, CopyDescriptor> &trans );
std::ostream &operator<<( std::ostream &os,
                          const DMATransfer<CopyDescriptor, void *> &trans );

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OpenCL_CACHE
