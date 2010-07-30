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

#include <iostream>
#include <exception>
#include "config.hpp"
#include "compatibility.hpp"
#include "instrumentor.hpp"
#include "system.hpp"
#include "directory.hpp"

namespace nanos {

  /*! \brief Represents a cache entry identified by an address
   */
   class CacheEntry : public Entry
   {
      private:
         /**< Address identifier of the cache entry  */
         void *_addr;

         /**< Entry references counter  */
         unsigned int _refs;

         bool _dirty;

      public:
        /*! \brief Default constructor
         */
         CacheEntry(): Entry(), _addr( NULL ), _refs( 0 ), _dirty( false ) {}

        /*! \brief Constructor
         *  \param addr: address of the cache entry
         */
         CacheEntry( void *addr, uint64_t tag, unsigned int version, bool dirty ): Entry( tag, version ), _addr( addr ), _refs( 0 ), _dirty( dirty ) {}

        /*! \brief Copy constructor
         *  \param Another CacheEntry
         */
         CacheEntry( const CacheEntry &ce ): Entry( ce.getTag(), ce.getVersion() ), _addr( ce._addr ), _refs( ce._refs ), _dirty( false ) {}

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
            this->_refs = ce._refs;
            this->_dirty = ce._dirty;
            return *this;
         }

        /* \brief Returns the address identifier of the Cache Entry
         */
         void * getAddress() const
         { return _addr; }

        /* \brief Address setter
         */
         void setAddress( void *addr )
         { _addr = addr; }

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
   };


   class Cache
   {
      public:
         virtual ~Cache() { }
         virtual void * allocate( size_t size ) = 0;
         virtual CacheEntry& newEntry( uint64_t tag, unsigned int version, bool dirty ) = 0;
         virtual void deleteEntry( uint64_t tag, size_t size ) = 0;
         virtual CacheEntry* getEntry( uint64_t tag ) = 0;
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

         virtual void unregisterCacheAccess( uint64_t tag, size_t size ) = 0;

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
            DirectoryEntry *de = _directory.getEntry( tag );

            if ( de == NULL ) { // Memory access not registered in the directory
               // Create directory entry, if the access is output, own it
               _directory.unLock();
               de = &( _directory.newEntry( tag, 0, ( output ? &_cache : NULL ) ) );

               // Create cache entry (we assume it is not in the cache)
               CacheEntry& ce = _cache.newEntry( tag, 0, output ); 
               ce.increaseRefs();
               ce.setAddress( _cache.allocate( size ) );

               // Need to copy in ?
               if ( input ) {
                  _cache.copyDataToCache( tag, size );
               }
            } else {
               Cache *owner = de->getOwner();
               if ( owner != NULL ) {
                  // FIXME Is dirty we need to interact with the other cache
                  //fprintf(stderr, "hola\n");
                  CacheEntry *ce = _cache.getEntry( tag );
                  if ( ce == NULL ) {
                     
                     if ( output ) {
                        de->setOwner( &_cache );
                        de->setVersion( de->getVersion() );
                     }

                     CacheEntry& nce = _cache.newEntry( tag, 0, output ); 
                     nce.increaseRefs();
                     nce.setAddress( _cache.allocate( size ) );

                     // Need to copy in ?
                     if ( input ) {
                        _cache.copyDataToCache( tag, size );
                     }
                  }
                  else
                  {
                     if ( ( ce->getVersion() < de->getVersion() ) && input ) {
                        _cache.copyDataToCache( tag, size );
                        ce->setVersion( de->getVersion() );
                     }
                     if ( output ) {
                        de->setOwner( &_cache );
                        ce->setDirty( true );
                        de->setVersion( de->getVersion() + 1 ) ;
                        ce->setVersion( de->getVersion() );
                     }

                  }
                  //fprintf(stderr, "end hola\n");

               } else {
                  // lookup in cache
                  CacheEntry *ce = _cache.getEntry( tag );
                  if ( ce != NULL ) {
                  //fprintf(stderr, "hola2\n");
                     if ( ( ce->getVersion() < de->getVersion() ) && input ) {
                        _cache.copyDataToCache( tag, size );
                        ce->setVersion( de->getVersion() );
                     }

                     if ( output ) {
                        de->setOwner( &_cache );
                        ce->setDirty( true );
                        de->setVersion( de->getVersion() + 1 ) ;
                        ce->setVersion( de->getVersion() );
                     }

                  //fprintf(stderr, "end hola2\n");
                     NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-hit") );
                     NANOS_INSTRUMENT( sys.getInstrumentor()->raisePointEvent( key, (nanos_event_value_t) tag ) );
                  } else {
                  //fprintf(stderr, "hola3\n");
                     if ( output ) {
                        de->setOwner( &_cache );
                        de->setVersion( de->getVersion() + 1 );
                     }
                     ce = & (_cache.newEntry( tag, de->getVersion(), output ) );
                     ce->increaseRefs();
                     ce->setAddress( _cache.allocate( size ) );
                     if ( input ) {
                        _cache.copyDataToCache( tag, size );
                     }
                  //fprintf(stderr, "end hola3\n");
                  }
                  ce->increaseRefs();
               }
            }

            _directory.unLock();
         }

         virtual void unregisterCacheAccess( uint64_t tag, size_t size )
         {
            CacheEntry *ce = _cache.getEntry( tag );
            //ensure (ce != NULL, "Cache has been corrupted");
            if ( ce->isDirty() ) {
               _cache.copyBackFromCache( tag, size );
               DirectoryEntry *de = _directory.getEntry( tag );
               //ensure (de != NULL, "Cache has been corrupted");
               de->setOwner( NULL );
               _directory.unLock();
            }
            ce->decreaseRefs();
         }

         virtual void registerPrivateAccess( uint64_t tag, size_t size, bool input, bool output )
         {
            // Private accesses are never found in the cache, the directory is not used because they can't be shared
            CacheEntry& ce = _cache.newEntry( tag, 0, output ); 
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
         /**< Maps keys with CacheEntries  */
         typedef TR1::unordered_map< uint64_t, CacheEntry> CacheHash;
         CacheHash _cache;
         Lock _lock;

         _Policy _policy;

         // disable copy constructor and assignment operator
         DeviceCache( const DeviceCache &cache );
         const DeviceCache & operator= ( const DeviceCache &cache );

      public:
        /* \brief Default constructor
         */
         DeviceCache() : _cache(), _policy( *this ) {}

         void * allocate( size_t size )
         {
            void *result;
            NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-malloc") );
            NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst( CACHE, key, (nanos_event_value_t) size) );
            _lock.acquire();
            result = _T::allocate( size );
            _lock.release();
            NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key ) );
            return result;
         }

         void deleteEntry( uint64_t tag, size_t size )
         {
            NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-free") );
            NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst ( CACHE, key, (nanos_event_value_t) size) );
            // it assumes the entry exists
            _lock.acquire();
            CacheEntry &ce = _cache[tag];
            _T::free( ce.getAddress() );
            _cache.erase( tag );
            _lock.release();
            NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key ) );
         }

        /* \brief get the Address in the cache for tag
         * \param tag: Identifier of the entry to look for
         */
         void * getAddress( uint64_t tag )
         {
            _lock.acquire();
            void *result = _cache[tag].getAddress();
            _lock.release();
            return result;
         }

        /* \brief Copy data from the address represented by the tag to the entry in the device.
         * \param tag: identifier of the entry
         * \param size: number of bytes to copy
         */
         void copyDataToCache( uint64_t tag, size_t size )
         {
            NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-copy-in") );
            NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst( MEM_TRANSFER, key, (nanos_event_value_t) size) );
            _lock.acquire();
            _T::copyIn( _cache[tag].getAddress(), tag, size );
            _lock.release();
            NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key ) );
         }

        /* \brief Copy back from the entry to the address represented by the tag.
         * \param tag: Entry identifier and address of original data
         * \param size: number of bytes to copy
         */
         void copyBackFromCache( uint64_t tag, size_t size )
         {
            NANOS_INSTRUMENT( static nanos_event_key_t key1 = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-copy-out") );
            NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst( MEM_TRANSFER, key1, size ) );
            _lock.acquire();
            CacheEntry &entry = _cache[tag];
            _T::copyOut( tag, entry.getAddress(), size );
            _lock.release();
            NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key1 ) );
         }

        /* \brief Perform local copy in the device for an entry
         * \param dst: Device destination address to copy to
         * \param tag: entry identifier to look for the source data
         * \param size: number of bytes to copy
         */
         void copyTo( void *dst, uint64_t tag, size_t size )
         {
            NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-local-copy") );
            NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst( MEM_TRANSFER, key, size ) );
            _lock.acquire();
            _T::copyLocal( dst, _cache[tag].getAddress(), size );
            _lock.release();
            NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key ) );
         }

         CacheEntry& newEntry( uint64_t tag, unsigned int version, bool dirty )
         {
            _lock.acquire();
            CacheEntry& ce = _cache[tag];
            _lock.release();
            ce.setTag( tag );
            ce.setVersion( version );
            ce.setDirty( dirty );
            return ce;
         }

         CacheEntry* getEntry( uint64_t tag )
         {
            _lock.acquire();
            CacheHash::iterator it = _cache.find( tag );
            if ( it == _cache.end() ) {
               _lock.release();
               return NULL;
            }
            CacheEntry& de = (*it).second;
            _lock.release();
            return &de;
         }

         void registerCacheAccess( uint64_t tag, size_t size, bool input, bool output )
         {
            _policy.registerCacheAccess( tag, size, input, output );
         }

         void unregisterCacheAccess( uint64_t tag, size_t size )
         {
            _policy.unregisterCacheAccess( tag, size );
         }

         void registerPrivateAccess( uint64_t tag, size_t size, bool input, bool output )
         {
            _policy.registerPrivateAccess( tag, size, input, output );
         }

         void unregisterPrivateAccess( uint64_t tag, size_t size )
         {
            _policy.unregisterPrivateAccess( tag, size );
         }
   };

   template <class _T, class _Policy = WriteThroughPolicy>
   class HostCache : public Cache
   {
     /* FIXME (see #195)
      *   - Code in the cache c file?
      *   - check for errors 
      */
      private:
         /**< Maps keys with CacheEntries  */
         typedef TR1::unordered_map< uint64_t, CacheEntry> CacheHash;
         CacheHash _cache;
         Lock _lock;

         _Policy _policy;

         // disable copy constructor and assignment operator
         HostCache( const HostCache &cache );
         const HostCache & operator= ( const HostCache &cache );

      public:
        /* \brief Default constructor
         */
         HostCache() : _cache(), _lock( Lock::FREE ), _policy( *this ) {}

         void * allocate( size_t size )
         {
            void *result = (void *) 0xdeadbeef;
            //fprintf(stderr, "hostcache: allocate %d\n", size);
            //NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-malloc") );
            //NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst( CACHE, key, (nanos_event_value_t) size) );
            //result = _T::allocate( size );
            //NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key ) );
            return result;
         }

         void deleteEntry( uint64_t tag, size_t size )
         {
            //NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-free") );
            //NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst ( CACHE, key, (nanos_event_value_t) size) );
            // it assumes the entry exists
            //CacheEntry &ce = _cache[tag];
            //_T::free( ce.getAddress() );
            _cache.erase( tag );
            //NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key ) );
         }

        /* \brief get the Address in the cache for tag
         * \param tag: Identifier of the entry to look for
         */
         void * getAddress( uint64_t tag )
         {
            //void *result = _cache[tag].getAddress();
            //return result;
            return (void *) tag;
         }

        /* \brief Copy data from the address represented by the tag to the entry in the device.
         * \param tag: identifier of the entry
         * \param size: number of bytes to copy
         */
         void copyDataToCache( uint64_t tag, size_t size )
         {
            //NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-copy-in") );
            //NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst( MEM_TRANSFER, key, (nanos_event_value_t) size) );
            //_T::copyIn( _cache[tag].getAddress(), tag, size );
            //NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key ) );
         }

        /* \brief Copy back from the entry to the address represented by the tag.
         * \param tag: Entry identifier and address of original data
         * \param size: number of bytes to copy
         */
         void copyBackFromCache( uint64_t tag, size_t size )
         {
            //NANOS_INSTRUMENT( static nanos_event_key_t key1 = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-copy-out") );
            //NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst( MEM_TRANSFER, key1, size ) );
            //CacheEntry &entry = _cache[tag];
            //_T::copyOut( tag, entry.getAddress(), size );
            //NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key1 ) );
         }

        /* \brief Perform local copy in the device for an entry
         * \param dst: Device destination address to copy to
         * \param tag: entry identifier to look for the source data
         * \param size: number of bytes to copy
         */
         void copyTo( void *dst, uint64_t tag, size_t size )
         {
            //NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-local-copy") );
            //NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst( MEM_TRANSFER, key, size ) );
            //_T::copyLocal( dst, _cache[tag].getAddress(), size );
            //NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key ) );
         }

         CacheEntry& newEntry( uint64_t tag, unsigned int version, bool dirty )
         {
            //fprintf(stderr, "[%d] hostcache: newentry\n", sys.getNetwork()->getNodeNum());
            _lock.acquire();
            CacheEntry& ce = _cache[tag];
            try {
            ce = _cache[tag];
            }
            catch (std::exception &e)
            {
               fprintf(stderr, "omg!\n");
            }
            ce.setTag( tag );
            ce.setVersion( version );
            ce.setDirty( dirty );
            _lock.release();
            return ce;
         }

         CacheEntry* getEntry( uint64_t tag )
         {
            _lock.acquire();
            CacheHash::iterator it = _cache.find( tag );
            if ( it == _cache.end() ) {
               _lock.release();
               return NULL;
            }
            CacheEntry& de = (*it).second;
            _lock.release();
            return &de;
         }

         void registerCacheAccess( uint64_t tag, size_t size, bool input, bool output )
         {
            _policy.registerCacheAccess( tag, size, input, output );
         }

         void unregisterCacheAccess( uint64_t tag, size_t size )
         {
            _policy.unregisterCacheAccess( tag, size );
         }

         void registerPrivateAccess( uint64_t tag, size_t size, bool input, bool output )
         {
            _policy.registerPrivateAccess( tag, size, input, output );
         }

         void unregisterPrivateAccess( uint64_t tag, size_t size )
         {
            _policy.unregisterPrivateAccess( tag, size );
         }
   };

}

#endif
