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

#ifndef _NANOS_OCL_CACHE
#define _NANOS_OCL_CACHE

#include "basethread_decl.hpp"
#include "cache_decl.hpp"
#include "copydescriptor_decl.hpp"
#include "simpleallocator.hpp"
#include "system_decl.hpp"
#include "oclwrapper.hpp"
#include "oclutils.hpp"
#include "oclconfig.hpp"

#include <limits>
#include <queue>

#include <cassert>

namespace nanos {
namespace ext {

class OCLAdapter;
class OCLProcessor;
class OCLCache;

class Size
{
public:
   static const size_t OperationMask;
   static const size_t HostSizeMask;

   static const size_t BufferIdMask;
   static const size_t DeviceSizeMask;
   static const size_t DeviceLocalSizeMask;

public:
   Size( size_t size ) : _size( size ) { }

public:
   size_t getAllocSize() const
   {
      return _size & (isHostOperation() ? HostSizeMask : DeviceSizeMask);
   }

   size_t getLocalAllocSize() const
   {
      assert( !isHostOperation() && "Local buffers allocated on device" );

      return _size & DeviceLocalSizeMask;
   }

   unsigned getId() const
   {
      assert( !isHostOperation() && "Host operations haven't Ids" );

      return (_size & BufferIdMask) >> 1;
   }

   size_t getRaw() const { return _size; }

public:
   bool isHostOperation() const { return !(_size & OperationMask); }
   bool isDeviceOperation() const { return _size & OperationMask; }

public:
   void dump();

private:
   size_t _size;
};

class Address
{
public:
   Address( void *addr,
            OCLCache &cache ) : _addr( addr ),
                                _cache( cache ) { }
public:
   void *getAllocAddr() const { return _addr; }

   void *getRaw() const { return _addr; }

public:
   inline bool isHostOperation() const;

private:
   void *_addr;
   OCLCache &_cache;
};

class OCLMemDump
{
protected:
   static Lock _dumpLock;

protected:
   void dumpCopyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size );
   void dumpCopyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size );

   void dumpCopyIn( void *localDst, CopyDescriptor &remoteSrc, Size size );
   void dumpCopyOut( CopyDescriptor &remoteDst, void *localSrc, Size size );
};

class OCLCache : public OCLMemDump
{
public:
  OCLCache(OCLAdapter &oclAdapter) : _hostCacheSize( 0 ),
                                     _devCacheSize( 0 ),
                                     _oclAdapter( oclAdapter ) { }

  OCLCache( const OCLCache &cache ); // Do not implement.
  const OCLCache &operator=( const OCLCache &cache ); // Do not implement.

public:
   void initialize();

   void *allocate( size_t size )
   {
      return allocate( Size( size ) );
   }

   void *reallocate( void *addr, size_t size, size_t ceSize )
   {
      return reallocate( Address( addr, *this ), Size( size ), Size( ceSize ) );
   }

   void free( void *addr )
   {
      free( Address( addr, *this ) );
   }

   bool copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size )
   {
      return copyIn( localDst, remoteSrc, Size( size ) );
   }

   bool copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size )
   {
      return copyOut( remoteDst, localSrc, Size (size) );
   }

public:
   void *getHostBase()
   {
      return reinterpret_cast<void *>( _hostAllocator.getBaseAddress() );
   }

   void *getDeviceBase()
   {
      return reinterpret_cast<void *>( _devAllocator.getBaseAddress() );
   }

   size_t getSize() const { return _devCacheSize + _hostCacheSize; }

   cl_mem toMemoryObj( unsigned id ) { return _bufIdMappings[id]; }
   
   cl_mem toMemoryObjSS( void * id ) { return _devBufAddrMappings[id]; }
   
   cl_mem toMemoryObjSizeSS( size_t size , void* addr);

private:
   void *allocate( Size size )
   {
      bool starSSMode=OCLConfig::getStarSSMode();
      if( !starSSMode && size.isHostOperation() )
         return hostAllocate( size );
      else
         return deviceAllocate( size );
   }

   void *hostAllocate( Size size );
   void *deviceAllocate( Size size );

   void *reallocate( Address addr, Size size, Size ceSize )
   {
      bool starSSMode=OCLConfig::getStarSSMode();
      if(!starSSMode && size.isHostOperation() && ceSize.isHostOperation() )
         return hostReallocate( addr, size, ceSize );
      else if(starSSMode || (size.isDeviceOperation() && ceSize.isDeviceOperation()) )
         return deviceReallocate( addr, size, ceSize );

      fatal( "Cannot realloc in different address space" );
   }

   void *hostReallocate( Address addr, Size size, Size ceSize );
   void *deviceReallocate( Address addr, Size size, Size ceSize );

   void free( Address addr )
   {
      bool starSSMode=OCLConfig::getStarSSMode();
      if( !starSSMode && addr.isHostOperation() )
         hostFree( addr );
      else
         deviceFree( addr );
   }

   void hostFree( Address addr );
   void deviceFree( Address addr );

   bool copyIn( void *localDst, CopyDescriptor &remoteSrc, Size size )
   {
      bool starSSMode=OCLConfig::getStarSSMode();

      if( !starSSMode && size.isHostOperation())
         return hostCopyIn( localDst, remoteSrc, size );
      else
         return deviceCopyIn( localDst, remoteSrc, size );
   }

   bool hostCopyIn( void *localDst, CopyDescriptor &remoteSrc, Size size )
   {
      // The front-end must generate copy-in request in order to force the
      // cluster back-end to move data across the network. Thus the storage for
      // the host cache is managed implicitly by the cluster back-end. Here we
      // see host-accessible addresses.

      return true;
   }

   bool deviceCopyIn( void *localDst, CopyDescriptor &remoteSrc, Size size );

   bool copyOut( CopyDescriptor &remoteDst, void *localSrc, Size size )
   {
      bool starSSMode=OCLConfig::getStarSSMode();
      if( !starSSMode && size.isHostOperation())
         return hostCopyOut( remoteDst, localSrc, size );
      else
         return deviceCopyOut( remoteDst, localSrc, size );
   }

   bool hostCopyOut( CopyDescriptor &remoteDst, void *localSrc, Size size )
   {
      // The front-end must generate a copy-out request in order to force the
      // cluster back-end to move data across the network. Thus, the storage for
      // the host cache is managed implicitly by the cluster back-end. Here we
      // see host-accessible addresses.

      return true;
   }

   bool deviceCopyOut( CopyDescriptor &remoteDst, void *localSrc, Size size );

private:
   size_t _hostCacheSize;
   size_t _devCacheSize;

   SimpleAllocator _hostAllocator;
   SimpleAllocator _devAllocator;

   OCLAdapter &_oclAdapter;
   std::map<unsigned, cl_mem> _bufIdMappings;
   std::map<size_t, cl_mem> _allocBuffMappings;
   std::map<void *, cl_mem> _devBufAddrMappings;
   std::map<void *, cl_mem> _bufAddrMappings;
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

class OCLDMA : public OCLMemDump
{
public:
   OCLDMA( OCLProcessor &proc ) : _proc( proc ) { }

   OCLDMA( const OCLDMA &dma ); // Do not implement.
   OCLDMA &operator=(const OCLDMA &dma); // Do not implement.

public:
   bool copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size );
   bool copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size );

   void syncTransfer( uint64_t hostAddress );
   void execTransfers();

private:
   Lock _lock;

   DMAInTransferQueue _ins;
   DMAOutTransferQueue _outs;

   OCLProcessor &_proc;
};

inline bool Address::isHostOperation() const {
   return _cache.getHostBase() <= _addr && _addr < _cache.getDeviceBase();
}

std::ostream &operator<<( std::ostream &os, Size size );

std::ostream &operator<<( std::ostream &os,
                          const DMATransfer<void *, CopyDescriptor> &trans );
std::ostream &operator<<( std::ostream &os,
                          const DMATransfer<CopyDescriptor, void *> &trans );

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OCL_CACHE
