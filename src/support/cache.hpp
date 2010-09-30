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

#ifndef _NANOS_CACHE
#define _NANOS_CACHE

#include "config.hpp"
#include "compatibility.hpp"
#include "instrumentation.hpp"
#include "system.hpp"
#include "directory.hpp"
#include "atomic.hpp"

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

         /**< Entry references counter  */
         unsigned int _refs;

         volatile bool _dirty;
         volatile bool _copying;
         volatile bool _flushing;
         Atomic<unsigned int> _transfers;

      public:

        /*! \brief Default constructor
         */
         CacheEntry(): Entry(), _addr( NULL ), _size(0), _refs( 0 ), _dirty( false ), _copying(false), _flushing(false), _transfers(0) {}

        /*! \brief Constructor
         *  \param addr: address of the cache entry
         */
         CacheEntry( void *addr, size_t size, uint64_t tag, unsigned int version, bool dirty, bool copying ): Entry( tag, version ), _addr( addr ), _size(size), _refs( 0 ), _dirty( dirty ), _copying(copying), _flushing(false), _transfers(0) {}

        /*! \brief Copy constructor
         *  \param Another CacheEntry
         */
         CacheEntry( const CacheEntry &ce ): Entry( ce.getTag(), ce.getVersion() ), _addr( ce._addr ), _size( ce._size ), _refs( ce._refs ), _dirty( ce._dirty ), _copying(ce._copying), _flushing(false), _transfers(0) {}

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
            this->_refs = ce._refs;
            this->_dirty = ce._dirty;
            this->_copying = ce._copying;
            this->_flushing = ce._flushing;
            this->_transfers = ce._transfers;
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

        /* \brief Whether the Entry has references or not
         */
         bool hasRefs() const
         { return _refs > 0; }

        /* \brief Increase the references to the entry
         */
         void increaseRefs()
         { _refs++; }

        /* \brief Decrease the references to the entry
         */
         bool decreaseRefs()
         { return (--_refs) == 0; }

         bool isDirty()
         { return _dirty; }

         void setDirty( bool dirty )
         { _dirty = dirty; }

         bool isCopying() const
         { return _copying; }

         void setCopying( bool copying )
         { _copying = copying; }

         bool isFlushing()
         { return _flushing; }

         void setFlushing( bool flushing )
         { _flushing = flushing; }

         bool hasTransfers()
         { return _transfers.value() > 0; }

         void increaseTransfers()
         { _transfers++; }

         bool decreaseTransfers()
         {
            _transfers--;
            return hasTransfers();
         }
   };


   class Cache
   {
      public:
         virtual ~Cache() { }
         virtual void * allocate( size_t size ) = 0;
         virtual CacheEntry& newEntry( uint64_t tag, size_t size, unsigned int version, bool dirty ) = 0;
         virtual CacheEntry& insert( uint64_t tag, CacheEntry& ce, bool& inserted ) = 0;
         virtual void deleteEntry( uint64_t tag, size_t size ) = 0;
         virtual CacheEntry* getEntry( uint64_t tag ) = 0;
         virtual void addReference( uint64_t tag ) = 0;
         virtual void deleteReference( uint64_t tag ) = 0;
         virtual void copyDataToCache( uint64_t tag, size_t size ) = 0;
         virtual void copyBackFromCache( uint64_t tag, size_t size ) = 0;
         virtual void copyTo( void *dst, uint64_t tag, size_t size ) = 0;
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

         virtual void registerCacheAccess( uint64_t tag, size_t size, bool input, bool output ) = 0;

         virtual void unregisterCacheAccess( uint64_t tag, size_t size, bool output ) = 0;

         virtual void registerPrivateAccess( uint64_t tag, size_t size, bool input, bool output ) = 0;
         virtual void unregisterPrivateAccess( uint64_t tag, size_t size ) = 0;
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

         virtual void registerCacheAccess( uint64_t tag, size_t size, bool input, bool output )
         {
            // FIXME: Make sure that creation is atomic (locks in the list, you don't wan to lock the whole hash)
            DirectoryEntry *de = _directory.getEntry( tag );
            CacheEntry *ce;
            if ( de == NULL ) { // Memory access not registered in the directory
               bool inserted;
               DirectoryEntry d = DirectoryEntry( tag, 0, ( output ? &_cache : NULL ) ); // FIXME: insert accessible through directory
               de = &(_directory.insert( tag, d, inserted ));
               if (!inserted) {
                  if ( output ) {
                     de->setOwner(&_cache);
                  }
               }
                                                                                  //FIXME: insert accessible through cache, increases refs
               CacheEntry c =  CacheEntry( NULL, size, tag, 0, output, input );
               ce = &(_cache.insert( tag, c, inserted ));
               if (inserted) { // allocate it
                  ce->setAddress( _cache.allocate(size) );
                  if (input) {
                     _cache.copyDataToCache( tag, size );
                  }
               } else {        // wait for address
                  while ( ce->getAddress() == NULL );
               }
            } else {
               // DirectoryEntry exists
               bool inserted = false;
               ce = _cache.getEntry( tag );
               if ( ce == NULL ) {
                  // Create a new CacheEntry
                  CacheEntry c = CacheEntry( NULL, size, tag, 0, output, input );
                  ce = &(_cache.insert( tag, c, inserted ));
                  if (inserted) { // allocate it
                     Cache *owner = de->getOwner();
                     if ( owner != NULL && !(!input && output) ) {
                        // FIXME: interaction with other caches, forcing copy back
                        // owner->invalidate(tag);
                        // CacheEntry *owners = owner->getEntry( tag );
                        // while( owners->isFlushing() );
                     }
                     ce->setAddress( _cache.allocate(size) );
                     if (input) {
                        while ( de->getOwner() != NULL );
                        _cache.copyDataToCache( tag, size );
                     }
                     if (output) {
                        de->setOwner(&_cache);
                        de->increaseVersion();
                        ce->increaseVersion();
                     }
                  } else {        // wait for address
                     // has to be input, otherwise the program is incorrect so just wait the address to exist
                     while ( ce->getAddress() == NULL );
                     _cache.addReference(tag);
                  }
               } else {
                  if ( de->getVersion() != ce->getVersion()) {
                     if ( ce->setVersionCS( de->getVersion()) ) {
                        Cache *owner = de->getOwner();
                        if ( owner != NULL && !(!input && output) ) {
                           // FIXME: interaction with other caches, forcing copy back
                           // owner->invalidate(tag);
                           // CacheEntry *owners = owner->getEntry( tag );
                           // while( owners->isFlushing() );
                        }
                        if (input) {
                           _cache.copyDataToCache( tag, size );
                        }
                     }
                  }
                  if (output) {
                     de->setOwner(&_cache);
                     de->increaseVersion();
                     ce->increaseVersion();
                  }
               }
            }
         }

         virtual void unregisterCacheAccess( uint64_t tag, size_t size, bool output )
         {
            CacheEntry *ce = _cache.getEntry( tag );
            // There's two reference deleting calls because getEntry places one reference
            _cache.deleteReference(tag);
            // FIXME: This could be optimized by keeping the entries referenced untill they are flushed,
            // another option would be to wait to flushes when freeing memory is needed
            _cache.deleteReference(tag);
            if ( output ) {
               _cache.copyBackFromCache( tag, size );
               ce->setFlushing(true);
            }
         }

         virtual void registerPrivateAccess( uint64_t tag, size_t size, bool input, bool output )
         {
            bool inserted;
            CacheEntry c =  CacheEntry( NULL, size, tag, 0, output, input );
            CacheEntry& ce = _cache.insert( tag, c, inserted );
            ensure ( inserted, "Private access cannot hit the cache.");
            ce.increaseRefs();
            ce.setAddress( _cache.allocate( size ) );
            if ( input )
               _cache.copyDataToCache( tag, size );
         }

         virtual void unregisterPrivateAccess( uint64_t tag, size_t size )
         {
            CacheEntry *ce = _cache.getEntry( tag );
            ensure ( ce != NULL, "Private access cannot miss in the cache.");
            if ( ce->isDirty() )
               _cache.copyBackFromCache( tag, size );
            _cache.deleteEntry( tag, size );
         }
   };

  /*! \brief A Cache is a class that provides basic services for registering and
   *         searching for memory blocks in a device using an identifier represented
   *         by an unsigned int of 64 bits which represents the address of the original
   *         data in the host. 
   */
   template <class _T, class _Policy = WriteThroughPolicy>
   class DeviceCache : public Cache
   {
     /* FIXME (see #195)
      *   - Code in the cache c file?
      *   - check for errors 
      */
      private:
         Directory &_directory;

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
         DeviceCache( size_t size ) : _directory( sys.getDirectory() ), _cache(), _policy( *this ), _size( size ), _usedSize(0) {}

         size_t getSize()
         { return _size; }

         void setSize( size_t size )
         { _size = size; }

         void * allocate( size_t size )
         {
            void *result;
            NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-malloc") );
            if ( _usedSize + size <= _size ) {
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_CACHE, key, (nanos_event_value_t) size) );
               result = _T::allocate( size );
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
            } else {
               CacheHash::KeyList kl;
               // FIXME: lock the cache
               _cache.listUnreferencedKeys( kl );
               CacheHash::KeyList::iterator it;
               for ( it = kl.begin(); it != kl.end(); it++ ) {
                  // Copy the entry because once erased it can be recycled
                  CacheEntry ce = *(_cache.find( it->second ));
                  if ( _cache.erase( it->second ) ) {
                     // FIXME: With writeback it will be necesary to copy back
                     _T::free( ce.getAddress() );
                     _usedSize -= ce.getSize();
                     if ( _usedSize + size <= _size )
                        break;
                  }
               }
               // FIXME: unlock
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_CACHE, key, (nanos_event_value_t) size) );
               result = _T::allocate( size );
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
            }
            _usedSize+= size;
            return result;
         }

         void deleteEntry( uint64_t tag, size_t size )
         {
            NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-free") );
            NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst ( NANOS_CACHE, key, (nanos_event_value_t) size) );
            // it assumes the entry exists
            CacheEntry &ce = _cache[tag];
            _T::free( ce.getAddress() );
            _usedSize -= ce.getSize();
            _cache.erase( tag );
            NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
         }

        /* \brief get the Address in the cache for tag
         * \param tag: Identifier of the entry to look for
         */
         void * getAddress( uint64_t tag )
         {
            void *result = _cache[tag].getAddress();
            return result;
         }

        /* \brief Copy data from the address represented by the tag to the entry in the device.
         * \param tag: identifier of the entry
         * \param size: number of bytes to copy
         */
         void copyDataToCache( uint64_t tag, size_t size )
         {
            NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-in") );
            NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER, key, (nanos_event_value_t) size) );
            _T::copyIn( _cache[tag].getAddress(), tag, size );
            NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
         }

        /* \brief Copy back from the entry to the address represented by the tag.
         * \param tag: Entry identifier and address of original data
         * \param size: number of bytes to copy
         */
         void copyBackFromCache( uint64_t tag, size_t size )
         {
            NANOS_INSTRUMENT( static nanos_event_key_t key1 = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );
            NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER, key1, size ) );
            CacheEntry &entry = _cache[tag];
            _T::copyOut( tag, entry.getAddress(), size );
            NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key1 ) );
         }

        /* \brief Perform local copy in the device for an entry
         * \param dst: Device destination address to copy to
         * \param tag: entry identifier to look for the source data
         * \param size: number of bytes to copy
         */
         void copyTo( void *dst, uint64_t tag, size_t size )
         {
            NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-local-copy") );
            NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER, key, size ) );
            _T::copyLocal( dst, _cache[tag].getAddress(), size );
            NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
         }

         CacheEntry& newEntry( uint64_t tag, size_t size, unsigned int version, bool dirty )
         {
            CacheEntry& ce = _cache[tag];
            ce.setTag( tag );
            ce.setSize( size );
            ce.setVersion( version );
            ce.setDirty( dirty );
            return ce;
         }

         CacheEntry& insert( uint64_t tag, CacheEntry& ce, bool& inserted )
         {
            return _cache.insert( tag, ce, inserted );
         }

         CacheEntry* getEntry( uint64_t tag )
         {
            return _cache.findAndReference( tag );
         }

         void addReference( uint64_t tag )
         {
            _cache.findAndReference(tag);
         }

         void deleteReference( uint64_t tag )
         {
            _cache.deleteReference(tag);
         }

         void registerCacheAccess( uint64_t tag, size_t size, bool input, bool output )
         {
            _policy.registerCacheAccess( tag, size, input, output );
         }

         void unregisterCacheAccess( uint64_t tag, size_t size, bool output )
         {
            _policy.unregisterCacheAccess( tag, size, output );
         }

         void registerPrivateAccess( uint64_t tag, size_t size, bool input, bool output )
         {
            _policy.registerPrivateAccess( tag, size, input, output );
         }

         void unregisterPrivateAccess( uint64_t tag, size_t size )
         {
            _policy.unregisterPrivateAccess( tag, size );
         }

         void synchronizeTransfer( uint64_t tag )
         {
            CacheEntry *ce = _cache.find(tag);
            // FIXME: enable this when separating the headers
            //ensure( ce != NULL && ce->hasTransfers(), "Cache has been corrupted" );
            ce->decreaseTransfers();
         }

         void synchronize( uint64_t tag )
         {
            CacheEntry *ce = _cache.find(tag);
            // FIXME: enable this when separating the headers
            //ensure( ce != NULL, "Cache has been corrupted" );
            if ( ce->isFlushing() ) {
               ce->setFlushing(false);
               DirectoryEntry *de = _directory.getEntry(tag);
               // FIXME: enable this when separating the headers
               //ensure( de != NULL, "Directory has been corrupted" );
               de->setOwner(NULL);
            } else {
               // FIXME: enable this when separating the headers
               //ensure( ce->isCopying(), "Cache has been corrupted" );
               ce->setCopying(false);
            }
         }

         static void synchronize( DeviceCache* _this, uint64_t tag )
         {
            CacheEntry *ce = _this->_cache.find(tag);
            // FIXME: enable this when separating the headers
            //ensure( ce != NULL, "Cache has been corrupted" );
            if ( ce->isFlushing() ) {
               ce->setFlushing(false);
               DirectoryEntry *de = _this->_directory.getEntry(tag);
               // FIXME: enable this when separating the headers
               //ensure( de != NULL, "Directory has been corrupted" );
               de->setOwner(NULL);
            } else {
               // FIXME: enable this when separating the headers
               //ensure( ce->isCopying(), "Cache has been corrupted" );
               ce->setCopying(false);
            }
         }

         void synchronize( std::list<uint64_t> &tags )
         {
            for_each( tags.begin(), tags.end(), std :: bind1st( std :: ptr_fun ( synchronize ), this ) );
         }

         void waitInput( uint64_t tag )
         {
            CacheEntry *ce = _cache.find(tag);
            // FIXME: enable this when separating the headers
            //ensure( ce != NULL, "Cache has been corrupted" );
            while ( ce->isCopying() );
         }

         static void waitInput( DeviceCache* _this, uint64_t tag )
         {
            CacheEntry *ce = _this->_cache.find(tag);
            // FIXME: enable this when separating the headers
            //ensure( ce != NULL, "Cache has been corrupted" );
            while ( ce->isCopying() );
         }

         void waitInputs( std::list<uint64_t> &tags )
         {
            for_each( tags.begin(), tags.end(), std :: bind1st( std :: ptr_fun ( waitInput ), this ) );
            for_each( tags.begin(), tags.end(), waitInput );
         }

         size_t& getCacheSize()
         {
            return _size;
         }
   };

}

#endif
