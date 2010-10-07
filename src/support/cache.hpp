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
#include "cache_decl.hpp"
#include "directory.hpp"
#include "atomic.hpp"
#include "processingelement_fwd.hpp"

using namespace nanos;

inline void CachePolicy::registerCacheAccess( uint64_t tag, size_t size, bool input, bool output )
{
   DirectoryEntry *de = _directory.getEntry( tag );
   CacheEntry *ce;
   if ( de == NULL ) { // Memory access not registered in the directory
      bool inserted;
      DirectoryEntry d = DirectoryEntry( tag, 0, ( output ? &_cache : NULL ) );
      de = &(_directory.insert( tag, d, inserted ));
      if (!inserted) {
         if ( output ) {
            de->setOwner(&_cache);
            de->setInvalidated(false);
         }
      }

      CacheEntry c =  CacheEntry( NULL, size, tag, 0, output, input );
      ce = &(_cache.insert( tag, c, inserted ));
      if (inserted) { // allocate it
         ce->setAddress( _cache.allocate(size) );
         if (input) {
            if ( _cache.copyDataToCache( tag, size ) ) {
               ce->setCopying(false);
            }
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
               owner->invalidate( tag, size, de );
               while( de->getOwner() != NULL )
                  _cache.syncTransfer( tag );
            }
            ce->setAddress( _cache.allocate(size) );
            if (input) {
               while ( de->getOwner() != NULL );
               if ( _cache.copyDataToCache( tag, size ) ) {
                  ce->setCopying(false);
               }
            }
            if (output) {
               de->setOwner(&_cache);
               de->setInvalidated(false);
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
                  owner->invalidate( tag, size, de );
                  while( de->getOwner() != NULL )
                     _cache.syncTransfer( tag );
               }
               if (input) {
                  if ( _cache.copyDataToCache( tag, size ) ) {
                     ce->setCopying(false);
                  }
               }
            }
         }
         if (output) {
            de->setOwner(&_cache);
            de->setInvalidated(false);
            de->increaseVersion();
            ce->increaseVersion();
         }
      }
   }
}

inline void CachePolicy::registerPrivateAccess( uint64_t tag, size_t size, bool input, bool output )
{
   bool inserted;
   CacheEntry c =  CacheEntry( NULL, size, tag, 0, output, input );
   CacheEntry& ce = _cache.insert( tag, c, inserted );
   ensure ( inserted, "Private access cannot hit the cache.");
   ce.increaseRefs();
   ce.setAddress( _cache.allocate( size ) );
   if ( input ) {
      if ( _cache.copyDataToCache( tag, size ) ) {
         ce.setCopying(false);
      }
   }
}

inline void CachePolicy::unregisterPrivateAccess( uint64_t tag, size_t size )
{
   CacheEntry *ce = _cache.getEntry( tag );
   ensure ( ce != NULL, "Private access cannot miss in the cache.");
   // FIXME: to use this output it needs to be synchronized now or somewhere in case it is asynchronous
   if ( ce->isDirty() )
      _cache.copyBackFromCache( tag, size );
   _cache.deleteEntry( tag, size );
}

inline void WriteThroughPolicy::unregisterCacheAccess( uint64_t tag, size_t size, bool output )
{
   CacheEntry *ce = _cache.getEntry( tag );
   // There's two reference deleting calls because getEntry places one reference
   _cache.deleteReference(tag);
   _cache.deleteReference(tag);
   if ( output ) {
      if ( _cache.copyBackFromCache( tag, size ) ) {
         ce->setDirty(false);
         DirectoryEntry *de = _directory.getEntry(tag);
         ensure( de != NULL, "Directory has been corrupted" );
         de->setOwner(NULL);
      } else {
         ce->setFlushing(true);
         ce->setDirty(false);
      }
   }
}

inline void WriteBackPolicy::unregisterCacheAccess( uint64_t tag, size_t size, bool output )
{
   // There's two reference deleting calls because getEntry places one reference
   _cache.deleteReference(tag);
   _cache.deleteReference(tag);
}

 /*! \brief A Cache is a class that provides basic services for registering and
  *         searching for memory blocks in a device using an identifier represented
  *         by an unsigned int of 64 bits which represents the address of the original
  *         data in the host. 
  */
template <class _T, class _Policy>
inline size_t DeviceCache<_T,_Policy>::getSize()
   { return _size; }

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::setSize( size_t size )
   { _size = size; }

template <class _T, class _Policy>
inline void * DeviceCache<_T,_Policy>::allocate( size_t size )
{
   void *result;
   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-malloc") );
   if ( _usedSize + size <= _size ) {
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_CACHE, key, (nanos_event_value_t) size) );
      result = _T::allocate( size, _pe );
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
            // FIXME: this can be optimized by adding the flushing entries to a list and go to that list if not enough space was freed
            while ( ce.isFlushing() )
               _T::syncTransfer( (uint64_t)it->second, _pe );
            _T::free( ce.getAddress(), _pe );
            _usedSize -= ce.getSize();
            if ( _usedSize + size <= _size )
               break;
         }
      }
      // FIXME: unlock
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_CACHE, key, (nanos_event_value_t) size) );
      result = _T::allocate( size, _pe );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
   }
   _usedSize+= size;
   return result;
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::deleteEntry( uint64_t tag, size_t size )
{
   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-free") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst ( NANOS_CACHE, key, (nanos_event_value_t) size) );
   // it assumes the entry exists
   CacheEntry &ce = _cache[tag];
   _T::free( ce.getAddress(), _pe );
   _usedSize -= ce.getSize();
   _cache.erase( tag );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
}

template <class _T, class _Policy>
inline void * DeviceCache<_T,_Policy>::getAddress( uint64_t tag )
{
   void *result = _cache[tag].getAddress();
   return result;
}

template <class _T, class _Policy>
inline bool DeviceCache<_T,_Policy>::copyDataToCache( uint64_t tag, size_t size )
{
   bool result;
   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-in") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER, key, (nanos_event_value_t) size) );
   result = _T::copyIn( _cache[tag].getAddress(), tag, size, _pe );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
   return result;
}

template <class _T, class _Policy>
inline bool DeviceCache<_T,_Policy>::copyBackFromCache( uint64_t tag, size_t size )
{
   bool result;
   NANOS_INSTRUMENT( static nanos_event_key_t key1 = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER, key1, size ) );
   CacheEntry &entry = _cache[tag];
   result = _T::copyOut( tag, entry.getAddress(), size, _pe );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key1 ) );
   return result;
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::copyTo( void *dst, uint64_t tag, size_t size )
{
   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-local-copy") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER, key, size ) );
   _T::copyLocal( dst, _cache[tag].getAddress(), size, _pe );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
}

template <class _T, class _Policy>
inline CacheEntry& DeviceCache<_T,_Policy>::newEntry( uint64_t tag, size_t size, unsigned int version, bool dirty )
{
   CacheEntry& ce = _cache[tag];
   ce.setTag( tag );
   ce.setSize( size );
   ce.setVersion( version );
   ce.setDirty( dirty );
   return ce;
}

template <class _T, class _Policy>
inline CacheEntry& DeviceCache<_T,_Policy>::insert( uint64_t tag, CacheEntry& ce, bool& inserted )
{
   return _cache.insert( tag, ce, inserted );
}

template <class _T, class _Policy>
inline CacheEntry* DeviceCache<_T,_Policy>::getEntry( uint64_t tag )
{
   return _cache.findAndReference( tag );
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::addReference( uint64_t tag )
{
   _cache.findAndReference(tag);
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::deleteReference( uint64_t tag )
{
   _cache.deleteReference(tag);
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::registerCacheAccess( uint64_t tag, size_t size, bool input, bool output )
{
   _policy.registerCacheAccess( tag, size, input, output );
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::unregisterCacheAccess( uint64_t tag, size_t size, bool output )
{
   _policy.unregisterCacheAccess( tag, size, output );
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::registerPrivateAccess( uint64_t tag, size_t size, bool input, bool output )
{
   _policy.registerPrivateAccess( tag, size, input, output );
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::unregisterPrivateAccess( uint64_t tag, size_t size )
{
   _policy.unregisterPrivateAccess( tag, size );
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::synchronizeTransfer( uint64_t tag )
{
   CacheEntry *ce = _cache.find(tag);
   ensure( ce != NULL && ce->hasTransfers(), "Cache has been corrupted" );
   ce->decreaseTransfers();
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::synchronize( uint64_t tag )
{
   CacheEntry *ce = _cache.find(tag);
   ensure( ce != NULL, "Cache has been corrupted" );
   if ( ce->isFlushing() ) {
      ce->setFlushing(false);
      DirectoryEntry *de = _directory.getEntry(tag);
      ensure( de != NULL, "Directory has been corrupted" );
      de->setOwner(NULL);
   } else {
      ensure( ce->isCopying(), "Cache has been corrupted" );
      ce->setCopying(false);
   }
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::synchronize( DeviceCache<_T,_Policy>* _this, uint64_t tag )
{
   CacheEntry *ce = _this->_cache.find(tag);
   ensure( ce != NULL, "Cache has been corrupted" );
   if ( ce->isFlushing() ) {
      ce->setFlushing(false);
      DirectoryEntry *de = _this->_directory.getEntry(tag);
      ensure( de != NULL, "Directory has been corrupted" );
      de->setOwner(NULL);
   } else {
      ensure( ce->isCopying(), "Cache has been corrupted" );
      ce->setCopying(false);
   }
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::synchronize( std::list<uint64_t> &tags )
{
   for_each( tags.begin(), tags.end(), std :: bind1st( std :: ptr_fun ( synchronize ), this ) );
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::waitInput( uint64_t tag )
{
   CacheEntry *ce = _cache.find(tag);
   ensure( ce != NULL, "Cache has been corrupted" );
   while ( ce->isCopying() );
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::waitInput( DeviceCache<_T,_Policy>* _this, uint64_t tag )
{
   CacheEntry *ce = _this->_cache.find(tag);
   ensure( ce != NULL, "Cache has been corrupted" );
   while ( ce->isCopying() );
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::waitInputs( std::list<uint64_t> &tags )
{
   for_each( tags.begin(), tags.end(), std :: bind1st( std :: ptr_fun ( waitInput ), this ) );
   for_each( tags.begin(), tags.end(), waitInput );
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::invalidate( uint64_t tag, size_t size, DirectoryEntry *de )
{
   CacheEntry *ce = _cache.find( tag );
   if ( de->trySetInvalidated() ) {
      if ( ce->trySetToFlushing() ) {
         if ( copyBackFromCache( tag, size ) ) {
            ce->setFlushing(false);
            de->setOwner(NULL);
         }
      }
   }
}

template <class _T, class _Policy>
inline size_t& DeviceCache<_T,_Policy>::getCacheSize()
{
   return _size;
}

template <class _T, class _Policy>
inline void DeviceCache<_T,_Policy>::syncTransfer( uint64_t tag ) {
   _T::syncTransfer( tag, _pe );
}

#endif
