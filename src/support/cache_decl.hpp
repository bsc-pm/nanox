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

#ifndef _NANOS_CACHE_DECL
#define _NANOS_CACHE_DECL

#include "config.hpp"
#include "compatibility.hpp"
#include "system.hpp"
#include "directory_decl.hpp"
#include "atomic.hpp"
#include "processingelement_fwd.hpp"

namespace nanos {

  /*! \brief Represents a cache entry identified by an address
   */
   class CacheEntry : public Entry
   {
      private:
         /**< Address identifier of the cache entry  */
         void *_addr;

         /**< Size of the block in the cache */
         size_t _size;
         /**< Size of the block allocated in the device */
         size_t _allocSize;

         volatile bool _dirty;
         Atomic<bool> _copying;
         Atomic<bool> _flushing;
         Atomic<unsigned int> _transfers;
         Atomic<bool> _resizing;

      public:

        /*! \brief Default constructor
         */
         CacheEntry(): Entry(), _addr( NULL ), _size(0), _allocSize(0), _dirty( false ), _copying(false), _flushing(false), _transfers(0), _resizing(false) {}

        /*! \brief Constructor
         *  \param addr: address of the cache entry
         */
         CacheEntry( void *addr, size_t size, uint64_t tag, unsigned int version, bool dirty, bool copying ): Entry( tag, version ), _addr( addr ), _size(size), _allocSize(0), _dirty( dirty ), _copying(copying), _flushing(false), _transfers(0), _resizing(false) {}

        /*! \brief Copy constructor
         *  \param Another CacheEntry
         */
         CacheEntry( const CacheEntry &ce ): Entry( ce.getTag(), ce.getVersion() ), _addr( ce._addr ), _size( ce._size ), _allocSize( ce._allocSize ), _dirty( ce._dirty ), _copying(ce._copying), _flushing(false), _transfers(0), _resizing(false) {}

        /* \brief Destructor
         */
         ~CacheEntry() {}

        /* \brief Assign operator
         */
         CacheEntry& operator=( const CacheEntry &ce )
         {
            if ( this == &ce ) return *this;
            this->setTag( ce.getTag() );
            this->setVersion( ce.getVersion() );
            this->_addr = ce._addr;
            this->_size = ce._size;
            this->_dirty = ce._dirty;
            this->_copying = ce._copying;
            this->_flushing = ce._flushing;
            this->_transfers = ce._transfers;
            this->_resizing = ce._resizing;
            return *this;
         }

        /* \brief Returns the address identifier of the Cache Entry
         */
         void * getAddress()
         { return _addr; }

        /* \brief Address setter
         */
         void setAddress( void *addr )
         { _addr = addr; }

        /* \brief Returns the size of the block in the cache
         */
         size_t getSize() const
         { return _size; }

        /* \brief Size setter
         */
         void setSize( size_t size )
         { _size = size; }

        /* \brief Returns the size of the block in the device
         */
         size_t getAllocSize() const
         { return _allocSize; }

        /* \brief Size setter
         */
         void setAllocSize( size_t size )
         { _allocSize = size; }

         bool isDirty()
         { return _dirty; }

         void setDirty( bool dirty )
         { _dirty = dirty; }

         bool isCopying() const
         { return _copying.value(); }

         void setCopying( bool copying )
         { _copying = copying; }

         bool trySetToCopying()
         {
            Atomic<bool> expected = false;
            Atomic<bool> value = true;
            return _flushing.cswap( expected, value );
         }

         bool isFlushing()
         { return _flushing.value(); }

         void setFlushing( bool flushing )
         { _flushing = flushing; }

         bool trySetToFlushing()
         {
            Atomic<bool> expected = false;
            Atomic<bool> value = true;
            return _flushing.cswap( expected, value );
         }

         bool hasTransfers()
         { return _transfers.value() > 0; }

         void increaseTransfers()
         { _transfers++; }

         bool decreaseTransfers()
         {
            _transfers--;
            return hasTransfers();
         }

         bool isResizing()
         {
            return _resizing.value();
         }

         void setResizing( bool resizing )
         {
            _resizing = resizing;
         }

         bool trySetToResizing()
         {
            Atomic<bool> expected = false;
            Atomic<bool> value = true;
            return _resizing.cswap( expected, value );
         }
   };


   class Cache
   {
      public:
         virtual ~Cache() { }
         virtual void * allocate( size_t size ) = 0;
         virtual void realloc( CacheEntry * ce, size_t size ) = 0;
         virtual CacheEntry& newEntry( uint64_t tag, size_t size, unsigned int version, bool dirty ) = 0;
         virtual CacheEntry& insert( uint64_t tag, CacheEntry& ce, bool& inserted ) = 0;
         virtual void deleteEntry( uint64_t tag, size_t size ) = 0;
         virtual CacheEntry* getEntry( uint64_t tag ) = 0;
         virtual void addReference( uint64_t tag ) = 0;
         virtual void deleteReference( uint64_t tag ) = 0;
         virtual bool copyDataToCache( uint64_t tag, size_t size ) = 0;
         virtual bool copyBackFromCache( uint64_t tag, size_t size ) = 0;
         virtual void copyTo( void *dst, uint64_t tag, size_t size ) = 0;
         virtual void invalidate( uint64_t tag, size_t size, DirectoryEntry *de ) = 0;
         virtual void syncTransfer( uint64_t tag ) = 0;
         virtual int getReferences( unsigned int tag ) = 0;
   };

   class CachePolicy
   {
      private:
         CachePolicy( const CachePolicy &policy );
         const CachePolicy & operator= ( const CachePolicy &policy );

      public:
         Cache& _cache;
         Directory& _directory;

         CachePolicy( Cache& cache ) : _cache( cache ), _directory( sys.getDirectory() ) { }

         virtual ~CachePolicy() { }

         virtual void registerCacheAccess( uint64_t tag, size_t size, bool input, bool output );

         virtual void unregisterCacheAccess( uint64_t tag, size_t size, bool output ) = 0;

         virtual void registerPrivateAccess( uint64_t tag, size_t size, bool input, bool output );

         virtual void unregisterPrivateAccess( uint64_t tag, size_t size );
   };

   // A plugin maybe??
   class WriteThroughPolicy : public CachePolicy
   {
      private:

         WriteThroughPolicy( const WriteThroughPolicy &policy );
         const WriteThroughPolicy & operator= ( const WriteThroughPolicy &policy );

      public:

         WriteThroughPolicy( Cache& cache ) : CachePolicy( cache ) { }

         virtual ~WriteThroughPolicy() { }

         virtual void unregisterCacheAccess( uint64_t tag, size_t size, bool output );
   };

   class WriteBackPolicy : public CachePolicy
   {
      private:

         WriteBackPolicy( const WriteBackPolicy &policy );
         const WriteBackPolicy & operator= ( const WriteBackPolicy &policy );

      public:

         WriteBackPolicy( Cache& cache ) : CachePolicy( cache ) { }

         virtual ~WriteBackPolicy() { }

         virtual void unregisterCacheAccess( uint64_t tag, size_t size, bool output );
   };

  /*! \brief A Cache is a class that provides basic services for registering and
   *         searching for memory blocks in a device using an identifier represented
   *         by an unsigned int of 64 bits which represents the address of the original
   *         data in the host. 
   */
   template <class _T, class _Policy = WriteThroughPolicy>
   //template <class _T, class _Policy = WriteBackPolicy>
   class DeviceCache : public Cache
   {
     /* FIXME (see #195)
      *   - Code in the cache c file?
      *   - check for errors 
      */
      private:
         Directory &_directory;
         ProcessingElement *_pe;

         /**< Maps keys with CacheEntries  */
         typedef HashMap<uint64_t, CacheEntry> CacheHash;
         CacheHash _cache;

         _Policy _policy;

         size_t _size;
         size_t _usedSize;

         // disable copy constructor and assignment operator
         DeviceCache( const DeviceCache &cache );
         const DeviceCache & operator= ( const DeviceCache &cache );

      public:
        /* \brief Default constructor
         */
         DeviceCache( size_t size, ProcessingElement *pe = NULL ) : _directory( sys.getDirectory() ), _pe( pe ), _cache(), _policy( *this ), _size( size ), _usedSize(0) {}

         size_t getSize();

         void setSize( size_t size );

         void * allocate( size_t size );

         void freeSpaceToFit( size_t size );

         void deleteEntry( uint64_t tag, size_t size );

         void realloc( CacheEntry *ce, size_t size );

        /* \brief get the Address in the cache for tag
         * \param tag: Identifier of the entry to look for
         */
         void * getAddress( uint64_t tag );

        /* \brief Copy data from the address represented by the tag to the entry in the device.
         * \param tag: identifier of the entry
         * \param size: number of bytes to copy
         */
         bool copyDataToCache( uint64_t tag, size_t size );

        /* \brief Copy back from the entry to the address represented by the tag.
         * \param tag: Entry identifier and address of original data
         * \param size: number of bytes to copy
         */
         bool copyBackFromCache( uint64_t tag, size_t size );

        /* \brief Perform local copy in the device for an entry
         * \param dst: Device destination address to copy to
         * \param tag: entry identifier to look for the source data
         * \param size: number of bytes to copy
         */
         void copyTo( void *dst, uint64_t tag, size_t size );

         CacheEntry& newEntry( uint64_t tag, size_t size, unsigned int version, bool dirty );

         CacheEntry& insert( uint64_t tag, CacheEntry& ce, bool& inserted );

         CacheEntry* getEntry( uint64_t tag );

         void addReference( uint64_t tag );

         void deleteReference( uint64_t tag );

         void registerCacheAccess( uint64_t tag, size_t size, bool input, bool output );

         void unregisterCacheAccess( uint64_t tag, size_t size, bool output );

         void registerPrivateAccess( uint64_t tag, size_t size, bool input, bool output );

         void unregisterPrivateAccess( uint64_t tag, size_t size );

         void synchronizeTransfer( uint64_t tag );

         void synchronize( uint64_t tag );

         static void synchronize( DeviceCache* _this, uint64_t tag );

         void synchronize( std::list<uint64_t> &tags );

         void waitInput( uint64_t tag );

         static void waitInput( DeviceCache* _this, uint64_t tag );

         void waitInputs( std::list<uint64_t> &tags );

         void invalidate( uint64_t tag, size_t size, DirectoryEntry *de );

         size_t& getCacheSize();

         void syncTransfer( uint64_t tag );

         int getReferences( unsigned int tag );
   };

}

#endif
