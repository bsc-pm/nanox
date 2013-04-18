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
#include "instrumentationmodule_decl.hpp"
#include "cache_decl.hpp"
#include "directory.hpp"
#include "atomic.hpp"
#include "processingelement_fwd.hpp"
#include "copydescriptor.hpp"

using namespace nanos;

typedef enum {
   NANOS_CACHE_EVENT_NULL_EVENT,
   NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_94,
   NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_112,
   NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_122,
   NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_141,
   NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_163,
   NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_185,
   NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_221,
   NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_239,
   NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_260,
   NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_292,
   NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_300,
   NANOS_CACHE_EVENT_FREE_SPACE_TO_FIT,
   NANOS_CACHE_EVENT_WAIT_INPUT,
   NANOS_CACHE_EVENT_GENERIC_EVENT
} cache_wait_event_value;
   


inline unsigned int Cache::getId() const
{
  return _id;
}

inline void CachePolicy::registerCacheAccess( Directory& dir, CopyData &cpdata, uint64_t tag )
{
   size_t size = cpdata.getSize();
   bool input = cpdata.isInput();
   bool output = cpdata.isOutput();
   bool didCopyIn = false;
   CacheEntry *ce;
   ce = _cache.getEntry( tag );
   unsigned int version=0;
   if ( ce != NULL ) version = ce->getVersion()+1;
   DirectoryEntry *de = dir.getEntry( tag, version );

   cpdata.cpDesc.tag = tag;
   cpdata.cpDesc.dirVersion = version;

   if ( de == NULL ) { // Memory access not registered in the directory
      bool inserted;
      DirectoryEntry d = DirectoryEntry( tag, 0, ( output ? &_cache : NULL ), dir.getCacheMapSize() );
      de = &(dir.insert( tag, d, inserted ));
      if (!inserted) {
         if ( output ) {
            de->setOwner(&_cache);
            de->setInvalidated(false);
            ce->setFlushTo( &dir );
         }
      }

      CacheEntry c =  CacheEntry( NULL, size, tag, 0, output, input );
      c.addReference();
      ce = &(_cache.insert( tag, c, inserted ));
      if (inserted) { // allocate it
         ce->setAddress( _cache.allocate( dir, size ) );
         ce->setAllocSize( size );
         if (input) {
            CopyDescriptor cd = CopyDescriptor( tag, 0, /* copying */ true, /* flushing */ false );
            ce->addTransfer( cd.dirVersion, INPUT, false );
            if ( _cache.copyDataToCache( cd, size ) ) {
               ce->finishTransfer( cd.dirVersion, INPUT, false );
            }
            cpdata.setCopyDescriptor( cd );
         }
      } else {        // wait for address
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_94 ); )
         while ( ce->getAddress() == NULL ) {}
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )
      }
   } else {
      // DirectoryEntry exists
      bool inserted = false;
      if ( ce == NULL ) {
         // Create a new CacheEntry
         CacheEntry c = CacheEntry( NULL, size, tag, 0, output, input );
         c.addReference();
         ce = &(_cache.insert( tag, c, inserted ));
         if ( inserted ) { // allocate it
            ce->setAddress( _cache.allocate( dir, size ) );
            ce->setAllocSize( size );
            Cache *owner = de->getOwner();
#ifdef NANOS_GPU_USE_CUDA32
            if ( owner != NULL && !(!input && output) ) {
               owner->invalidateAndFlush( dir, tag, size, de );
               owner->syncTransfer(tag);
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_112 ); )
               while( de->getOwner() != NULL ) myThread->idle();
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )
            }
#else
            if ( owner != NULL ) {
               // Most recent version is in another cache
               if ( output ) {
                  // I am going to write, so invalidate it from the other cache
                  // There may be other caches that have the data, so they should check the version at each access
                  owner->invalidate( dir, tag, de );
               }
               if ( input ) {
                  CopyDescriptor cd = CopyDescriptor( tag, 0, /* copying */ true, /* flushing */ false );
                  ce->addTransfer( cd.dirVersion, INPUT, false );
                  CacheEntry * ownerCE = owner->getEntry( tag );
                  // TODO: For asynchronous transfers, reference to the owner will never be decreased
                  ownerCE->addReference();
                  if ( _cache.copyData( ce->getAddress(), cd, ownerCE->getAddress(), size, *owner ) ) {
                     ce->finishTransfer( cd.dirVersion, INPUT, false );
                     ownerCE->deleteReference();
                     cd.copying = false;
                  }
                  cpdata.setCopyDescriptor( cd );
               }
            } else if ( input ) {
               CopyDescriptor cd = CopyDescriptor( tag, 0, /* copying */ true, /* flushing */ false );
               ce->addTransfer( cd.dirVersion, INPUT, false );
               if ( _cache.copyDataToCache( cd, size ) ) {
                  ce->finishTransfer( cd.dirVersion, INPUT, false );
                  cd.copying = false;
               }
               cpdata.setCopyDescriptor( cd );
            }
#endif

#ifdef NANOS_GPU_USE_CUDA32
            if (input) {
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_122 ); )
               while ( de->getOwner() != NULL ) myThread->idle();
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )
               CopyDescriptor cd = CopyDescriptor(tag);
               if ( _cache.copyDataToCache( cd, size ) ) {
                  ce->setCopying(false);
               }
               cpdata.setCopyDescriptor( cd );
            }
#endif

            if ( output ) {
               de->setOwner( &_cache );
               de->setInvalidated( false );
               de->increaseVersion();
               ce->setFlushTo( &dir );
            }
            ce->setVersion( de->getVersion() );
         } else {        // wait for address
            // has to be input, otherwise the program is incorrect so just wait the address to exist
            NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_141 ); )
            while ( ce->getAddress() == NULL ) {}
            NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )

            if ( size != ce->getSize() ) {
               if ( ce->trySetToResizing() ) {
                  // Wait until it's only me using this entry
                  // FIXME: Multiple threads per cache not supported with this implementation
                  // of resize (references must be at most two due to prefetch) (see #393)
                  ensure( _cache.getReferences( ce->getTag() ) <= 2, "Multiple threads per cache not supported with this implementation");
//                  while ( _cache.getReferences( ce->getTag() ) > 1 );

                  // First approach, always copy back if size didn't match
                  if ( ce->isDirty() ) {
                     // invalidate in its own cache
                     _cache.invalidateAndFlush( dir, tag, ce->getSize(), de );
                     // synchronize invalidation
                     _cache.syncTransfer( tag ); // Ask the device to be nice and prioritize this transfer
                     NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_163 ); )
                     while( de->getOwner() != NULL ) myThread->processTransfers(); //myThread->idle();
                     NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )

                  }
                  if ( size > ce->getAllocSize() ) {
                     _cache.realloc( dir, ce, size );
                  }
                  ce->setSize(size);

                  if ( input ) {
                     didCopyIn = true;
                     Cache *owner = de->getOwner();
                     ensure( &_cache != owner, "Trying to invalidate myself" );
                     if ( owner != NULL ) {
                        // Is dirty somewhere else, we need to invalidate 'tag' in 'cache' and wait for synchronization
                        owner->invalidateAndFlush( dir, tag, size, de );
                        owner->syncTransfer( tag ); // Ask the device to be nice and prioritize this transfer
                        NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_185 ); )
                        while( de->getOwner() != NULL ) myThread->processTransfers(); //myThread->idle();
                        NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )

                     }

                     // Copy in
                     CopyDescriptor cd = CopyDescriptor( tag, 0, /* copying */ true, /* flushing */ false );
                     ce->addTransfer( cd.dirVersion, INPUT, false );
                     if ( _cache.copyDataToCache( cd, size ) ) {
                        ce->finishTransfer( cd.dirVersion, INPUT, false );
                        cd.copying = false;
                     }
                     cpdata.setCopyDescriptor( cd );
                  }
                  ce->setResizing(false);
               }
            }
         }
      } else {
         ce->addReference();

         // Cache entry already exists in the cache
         if ( size != ce->getSize() ) {
            if ( ce->trySetToResizing() ) {
               // Wait until it's only me using this entry
               // FIXME: Multiple threads per cache not supported with this implementation
               // of resize (references must be at most two due to prefetch) (see #393)
               ensure( _cache.getReferences( ce->getTag() ) <= 2, "Multiple threads per cache not supported with this implementation");
//               while ( _cache.getReferences( ce->getTag() ) > 1 );

               // First approach, always copy back if size didn't match
               if ( ce->isDirty() ) {
                  // invalidate in its own cache
                  _cache.invalidateAndFlush( dir, tag, ce->getSize(), de );
                  // synchronize invalidation
                  _cache.syncTransfer( tag ); // Ask the device to be nice and prioritize this transfer
                  NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_221 ); )
                  while( de->getOwner() != NULL ) myThread->processTransfers(); //myThread->idle();
                  NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )

               }
               if ( size > ce->getAllocSize() ) {
                  _cache.realloc( dir, ce, size );
               }
               ce->setSize(size);
 
               if ( input ) {
                  didCopyIn = true;
                  if ( ce->isFlushing() ) {
                     Cache *owner = de->getOwner();
                     owner->syncTransfer( tag ); // Ask the device to be nice and prioritize this transfer
                     NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_239 ); )
                     while( de->getOwner() != NULL ) myThread->processTransfers(); //myThread->idle();
                     NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )
                     // Copy in
                     CopyDescriptor cd = CopyDescriptor( tag, 0, /* copying */ true, /* flushing */ false );
                     ce->addTransfer( cd.dirVersion, INPUT, false );
                     if ( _cache.copyDataToCache( cd, size ) ) {
                        ce->finishTransfer( cd.dirVersion, INPUT, false );
                        cd.copying = false;
                     }
                     cpdata.setCopyDescriptor( cd );
                  } else { 
                     Cache *owner = de->getOwner();
                     ensure( &_cache != owner, "Trying to invalidate myself" );
                     if ( owner != NULL ) {
                        // Is dirty somewhere else, we need to invalidate 'tag' in 'cache' and wait for synchronization
                        owner->invalidateAndFlush( dir, tag, size, de );
                        owner->syncTransfer( tag ); // Ask the device to be nice and prioritize this transfer
                        NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_260 ); )
                        while( de->getOwner() != NULL ) myThread->processTransfers(); //myThread->idle();
                        NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )

                     }

                     // Copy in
                     CopyDescriptor cd = CopyDescriptor( tag, 0, /* copying */ true, /* flushing */ false );
                     ce->addTransfer( cd.dirVersion, INPUT, false );
                     if ( _cache.copyDataToCache( cd, size ) ) {
                        ce->finishTransfer( cd.dirVersion, INPUT, false );
                        cd.copying = false;
                     }
                     cpdata.setCopyDescriptor( cd );
                  }
               }
               ce->setResizing(false);
            }
         }

         if ( de->getVersion() != ce->getVersion() ) {
            // Version doesn't match the one in the directory
            if ( input && !didCopyIn ) {
               ce->setVersion( de->getVersion() );
               Cache *owner = de->getOwner();
               ensure( &_cache != owner, "Trying to invalidate myself" );
#ifdef NANOS_GPU_USE_CUDA32
               if ( owner != NULL ) {
                  // Is dirty somewhere else, we need to invalidate 'tag' in 'cache' and wait for synchronization
                  owner->invalidateAndFlush( dir, tag, size, de );
                  owner->syncTransfer( tag ); // Ask the device to be nice and prioritize this transfer
                  NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_292 ); )
                  while( de->getOwner() != NULL ) myThread->idle();
                  NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )

               }

               // Wait while it's resizing
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_REGISTER_CACHE_ACCESS_300 ); )
               while ( ce->isResizing() ) {}
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )

               // Copy in
               CopyDescriptor cd = CopyDescriptor(tag);
               if ( _cache.copyDataToCache( cd, size ) ) {
                  ce->setCopying(false);
               }
               cpdata.setCopyDescriptor( cd );
#else

               if ( owner != NULL ) {
                  // Most recent version is in another cache
                  // It is input for sure, as we have checked it before
                  if ( output ) {
                     // I am going to write, so invalidate it from the other cache
                     // There may be other caches that have the data, so they should check the version at each access
                     owner->invalidate( dir, tag, de );
                  }

                  CopyDescriptor cd = CopyDescriptor( tag, 0, /* copying */ true, /* flushing */ false );
                  ce->addTransfer( cd.dirVersion, INPUT, false );
                  CacheEntry * ownerCE = owner->getEntry( tag );
                  // TODO: For asynchronous transfers, reference to the owner will never be decreased
                  ownerCE->addReference();
                  if ( _cache.copyData( ce->getAddress(), cd, ownerCE->getAddress(), size, *owner ) ) {
                     ce->finishTransfer( cd.dirVersion, INPUT, false );
                     ownerCE->deleteReference();
                     cd.copying = false;
                  }
                  cpdata.setCopyDescriptor( cd );

               } else {
                  CopyDescriptor cd = CopyDescriptor( tag, 0, /* copying */ true, /* flushing */ false );
                  ce->addTransfer( cd.dirVersion, INPUT, false );
                  if ( _cache.copyDataToCache( cd, size ) ) {
                     ce->finishTransfer( cd.dirVersion, INPUT, false );
                     cd.copying = false;
                  }
                  cpdata.setCopyDescriptor( cd );
               }
#endif
            } else {
               // Since there's no input, it is output and we don't care about what may be in other caches, just update this version
               ce->setVersion( de->getVersion() );
            }
         }
         if ( output ) {
            de->setOwner( &_cache );
            de->setInvalidated( false );
            de->increaseVersion();
            ce->increaseVersion();
            ce->setFlushTo( &dir );
            ensure( de->getVersion() == ce->getVersion(), "Version mismatch between cache and directory entry." );
         }
      }
   }

   de->addAccess( _cache.getId() );
}

inline void CachePolicy::registerPrivateAccess( Directory& dir, CopyData &cpdata, uint64_t tag )
{
   size_t size = cpdata.getSize();
   bool input = cpdata.isInput();
   bool output = cpdata.isOutput();
   bool inserted;
   CacheEntry c =  CacheEntry( NULL, size, tag, 0, output, input );
   CacheEntry& ce = _cache.insert( tag, c, inserted );
   ensure ( inserted, "Private access cannot hit the cache.");
   ce.setAddress( _cache.allocate( dir, size ) );
   ce.setAllocSize( size );
   if ( input ) {
      CopyDescriptor cd = CopyDescriptor( tag, 0, /* copying */ true, /* flushing */ false );
      ce.addTransfer( cd.dirVersion, INPUT, false );
      if ( _cache.copyDataToCache( cd, size ) ) {
         ce.finishTransfer( cd.dirVersion, INPUT, false );
         cd.copying = false;
      }
      cpdata.setCopyDescriptor( cd );
   }
}

inline void CachePolicy::unregisterPrivateAccess( Directory &dir, CopyData &cpdata, uint64_t tag )
{
   size_t size = cpdata.getSize();
   CacheEntry *ce = _cache.getEntry( tag );
   ensure ( ce != NULL, "Private access cannot miss in the cache.");
   // FIXME: to use this output it needs to be synchronized now or somewhere in case it is asynchronous
   if ( ce->isDirty() ) {
      CopyDescriptor cd = CopyDescriptor( tag, 0, /* copying */ false, /* flushing */ true );
      ce->addTransfer( cd.dirVersion, OUTPUT, true );
      ce->deleteReference();
      if ( _cache.copyBackFromCache( cd, size ) ) {
         ce->finishTransfer( cd.dirVersion, OUTPUT, true );
         cd.flushing = false;
         _cache.deleteEntry( tag, size );
      }
      cpdata.setCopyDescriptor( cd );
   } else {
      ce->deleteReference();
   }
}

inline void NoCache::registerCacheAccess( Directory& dir, CopyData &cpdata, uint64_t tag )
{
   size_t size = cpdata.getSize();
   bool input = cpdata.isInput();
   bool output = cpdata.isOutput();
   bool inserted;
   CacheEntry c =  CacheEntry( NULL, size, tag, 0, output, input );
   CacheEntry& ce = _cache.insert( tag, c, inserted );
   // TODO: The ensure is activated... why?
   //ensure ( inserted, "Private access cannot hit the cache.");

   if ( inserted ) {
      ce.setAddress( _cache.allocate( dir, size ) );
      ce.setAllocSize( size );
   }

   if ( input ) {
      CopyDescriptor cd = CopyDescriptor( tag, 0, /* copying */ true, /* flushing */ false );
      _cache.copyDataToCache( cd, size );
      cd.copying = false;
      cpdata.setCopyDescriptor( cd );
   }

}

inline void NoCache::unregisterCacheAccess( Directory& dir, CopyData &cpdata, uint64_t tag, bool output )
{
   size_t size = cpdata.getSize();
   if ( output ) {
      CopyDescriptor cd = CopyDescriptor( tag, 0, /* copying */ false, /* flushing */ true  );
      _cache.copyBackFromCache( cd, size );
      cd.flushing = false;
      cpdata.setCopyDescriptor( cd );
   }

   _cache.deleteReference( tag );

   if ( _cache.getReferences( tag ) == 0 ) {
      _cache.deleteEntry( tag, size );
   }
}

inline void NoCache::registerPrivateAccess( Directory& dir, CopyData &cpdata, uint64_t tag )
{
   registerCacheAccess( dir, cpdata, tag );
}

inline void NoCache::unregisterPrivateAccess( Directory &dir, CopyData &cpdata, uint64_t tag )
{
   CacheEntry *ce = _cache.getEntry( tag );
   ensure ( ce != NULL, "Private access cannot miss in the cache.");
   unregisterCacheAccess( dir, cpdata, tag, ce->isDirty() );
}

inline void WriteThroughPolicy::unregisterCacheAccess( Directory& dir, CopyData &cpdata, uint64_t tag, bool output )
{
   CacheEntry *ce = _cache.getEntry( tag );
   DirectoryEntry *de = dir.getEntry( tag );
   if ( output ) {
      ensure( de != NULL, "Directory has been corrupted" );
      CopyDescriptor cd = CopyDescriptor( tag, de->getVersion(), /* copying */ false, /* flushing */ true );
      ce->addTransfer( cd.dirVersion, OUTPUT, true );
      if ( _cache.copyBackFromCache( cd, cpdata.getSize() ) ) {
         ce->finishTransfer( cd.dirVersion, OUTPUT, true );
         cd.flushing = false;
         if ( !ce->isFlushing() ) {
            de->setOwner( NULL );
         }
      }
      cpdata.setCopyDescriptor( cd );
   }

   ce->deleteReference();

   if ( de != NULL ) {
      de->removeAccess( _cache.getId() );
   } else {
      warning("Directory entry not found at unregisterCacheAcces, this can be a problem.");
   }
}

inline void WriteBackPolicy::unregisterCacheAccess( Directory &dir, CopyData &cpdata, uint64_t tag, bool output )
{
   CacheEntry * ce = _cache.getEntry( tag );
   CopyDescriptor cd = CopyDescriptor( tag, ce->getVersion(), /* copying */ false, /* flushing */ false );
   cpdata.setCopyDescriptor( cd );
   ce->deleteReference();
}

inline Cache::Cache() : _id( sys.getCacheMap().registerCache() ) {}

template <class _T>
inline void DeviceCache<_T>::setPolicy( CachePolicy * policy )
{
   _policy = policy;
}

template <class _T>
inline size_t DeviceCache<_T>::getSize()
   { return _size; }

template <class _T>
ProcessingElement * DeviceCache<_T>::getPE()
{
   return _pe;
}

template <class _T>
inline void * DeviceCache<_T>::allocate( Directory &dir, size_t size )
{
   void *result;
   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-malloc") );
   if ( _usedSize + size <= _size ) {
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_CACHE, key, (nanos_event_value_t) size) );
      result = _T::allocate( size, _pe );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
   } else {
      // FIXME: lock the cache
      freeSpaceToFit( dir, size );
      // FIXME: unlock
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_CACHE, key, (nanos_event_value_t) size) );
      result = _T::allocate( size, _pe );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
   }
   _usedSize+= size;
   return result;
}

template <class _T>
inline void DeviceCache<_T>::freeSpaceToFit( Directory &dir, size_t size )
{
   std::list<CacheEntry *> entries;
   getUnreferencedEntries( entries );
   std::list<CacheEntry *>::iterator it;

   DirectoryEntry * de2 = NULL;
   for ( it = entries.begin(); it != entries.end(); it++ ) {
      de2 = NULL;
      CacheEntry *ce = *it;
      CopyDescriptor cd = CopyDescriptor( 0, 0, /* copying */ false, /* flushing */ false);
      if ( ce->isDirty() ) {
         DirectoryEntry *de = dir.getEntry( ce->getTag() );
         de2 = de;
         if ( de->getOwner() != this ) {
            // someone flushed it between setting to invalidated and setting to flushing, do nothing
         } else {
            // This CE is the owner
            cd.tag = ce->getTag();
            cd.dirVersion = de->getVersion();
            cd.copying = false;
            cd.flushing = true;
            ce->addTransfer( cd.dirVersion, OUTPUT, true );

            if ( copyBackFromCache( cd, ce->getSize() ) ) {
               ce->finishTransfer( cd.dirVersion, OUTPUT, true );
               cd.flushing = false;
               if ( !ce->isFlushing() ) {
                  de->setOwner( NULL );
               }
               de2 = NULL;
            }
         }
      }
      /* FIXME: this can be optimized by adding the flushing entries to a
       * list and go to that list if not enough space was freed
       */
      /* Wait loop:
       *  - requesting the transfer to the device.
       *  - idle must be done to allow the thread to manage the copies
       */
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_FREE_SPACE_TO_FIT ); )
      {

         myThread->disableGettingWork();
         while ( cd.tag != 0 && ce->isTransferPending( cd.dirVersion, OUTPUT, true ) ) {
            _T::syncTransfer( ce->getTag(), _pe );
            //myThread->idle();
            myThread->processTransfers();
         }

         if ( de2 != NULL && !ce->isFlushing() ) {
            de2->setOwner( NULL );
         }
         myThread->enableGettingWork();
      }
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )

      if ( !ce->isFlushing() ) {
         // Copy the entry because once erased it can be recycled
         CacheEntry ce2 = *ce;

         while ( _cache.getReferenceCount( ce2.getTag() ) != 0 ) {
            _cache.deleteReference( ce2.getTag() );
            _cache.deleteReference( ce2.getTag() );
         }

         if ( _cache.erase( ce2.getTag() ) ) {
            _T::free( ce2.getAddress(), _pe );
            _usedSize -= ce2.getSize();
            if ( _usedSize + size <= _size )
               break;
         }
      }
   }

   ensure( _usedSize + size <= _size, "Cache is full" );
}

template <class _T>
inline void DeviceCache<_T>::deleteEntry( uint64_t tag, size_t size )
{
   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-free") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst ( NANOS_CACHE, key, (nanos_event_value_t) size) );
   // it assumes the entry exists
   CacheEntry &ce = _cache[tag];
   _T::free( ce.getAddress(), _pe );
   _usedSize -= ce.getAllocSize();
   _cache.erase( tag );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
}

template <class _T>
inline void DeviceCache<_T>::realloc( Directory& dir, CacheEntry *ce, size_t size )
{
   if ( _usedSize + size - ce->getSize() < _size ) {
      freeSpaceToFit( dir, size - ce->getSize() );
   }
   _usedSize += size - ce->getSize();
   void *addr = ce->getAddress();
   ensure( size > ce->getSize() , "Trying to downsize a cache entry" );
   addr = _T::realloc( addr, size, ce->getSize(), _pe );
   ce->setAllocSize( size );
   ce->setSize( size );
   ce->setAddress( addr );
}

template <class _T>
inline void * DeviceCache<_T>::getAddress( uint64_t tag )
{
   void *result = _cache[tag].getAddress();
   return result;
}

template <class _T>
inline bool DeviceCache<_T>::copyData( void * dstAddr, CopyDescriptor &dstCd, void * srcAddr, size_t size, Cache & owner )
{
   bool result;
   DeviceCache< _T> *srcCache = dynamic_cast<DeviceCache< _T> *>( &owner );
   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-local-copy") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER_LOCAL, key, size ) );
   result = _T::copyDevToDev( dstAddr, dstCd, srcAddr, size, _pe, srcCache->getPE() );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
   return result;
}

template <class _T>
inline bool DeviceCache<_T>::copyDataToCache( CopyDescriptor &cd, size_t size )
{
   bool result;
   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-in") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER_IN, key, (nanos_event_value_t) size) );
   result = _T::copyIn( _cache[cd.getTag()].getAddress(), cd, size, _pe );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
   return result;
}

template <class _T>
inline bool DeviceCache<_T>::copyBackFromCache( CopyDescriptor &cd, size_t size )
{
   bool result;
   NANOS_INSTRUMENT( static nanos_event_key_t key1 = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER_OUT, key1, size ) );
   CacheEntry &entry = _cache[cd.getTag()];
   result = _T::copyOut( cd, entry.getAddress(), size, _pe );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key1 ) );
   return result;
}

template <class _T>
inline void DeviceCache<_T>::copyTo( void *dst, uint64_t tag, size_t size )
{
   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-local-copy") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER_LOCAL, key, size ) );
   _T::copyLocal( dst, _cache[tag].getAddress(), size, _pe );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
}

template <class _T>
inline CacheEntry& DeviceCache<_T>::newEntry( uint64_t tag, size_t size, unsigned int version, bool dirty )
{
   CacheEntry& ce = _cache[tag];
   ce.setTag( tag );
   ce.setSize( size );
   ce.setVersion( version );
   ce.setDirty( dirty );
   return ce;
}

template <class _T>
inline CacheEntry& DeviceCache<_T>::insert( uint64_t tag, CacheEntry& ce, bool& inserted )
{
   return _cache.insert( tag, ce, inserted );
}

template <class _T>
inline CacheEntry* DeviceCache<_T>::getEntry( uint64_t tag )
{
   CacheEntry * ce = _cache.find( tag );
   return ce;
}

template <class _T>
inline void DeviceCache<_T>::addReference( uint64_t tag )
{
   _cache.find( tag )->addReference();
}

template <class _T>
inline void DeviceCache<_T>::deleteReference( uint64_t tag )
{
   _cache.find( tag )->deleteReference();
}

template <class _T>
inline void DeviceCache<_T>::registerCacheAccess( Directory &dir, CopyData &cpdata, uint64_t tag )
{
   _policy->registerCacheAccess( dir, cpdata, tag );
}

template <class _T>
inline void DeviceCache<_T>::unregisterCacheAccess( Directory &dir, CopyData &cpdata, uint64_t tag, bool output )
{
   _policy->unregisterCacheAccess( dir, cpdata, tag, output );
}

template <class _T>
inline void DeviceCache<_T>::registerPrivateAccess( Directory &dir, CopyData &cpdata, uint64_t tag )
{
   _policy->registerPrivateAccess( dir, cpdata, tag );
}

template <class _T>
inline void DeviceCache<_T>::unregisterPrivateAccess( Directory &dir, CopyData &cpdata, uint64_t tag )
{
   _policy->unregisterPrivateAccess( dir, cpdata, tag );
}

template <class _T>
inline void DeviceCache<_T>::synchronizeInternal( SyncData &sd, CopyDescriptor &cd )
{
   if ( cd.copying && cd.flushing ) {
      return;
   }

   CacheEntry *ce = sd._this->_cache.find( cd.getTag() );
   ensure( ce != NULL, "Cache has been corrupted" );

   if ( cd.flushing ) {
      Directory* dir = ce->getFlushTo();
      ensure( dir != NULL, "CopyBack sync lost its directory");
      DirectoryEntry *de = dir->getEntry( cd.getTag() );
      //ensure ( !ce->isCopying(), "User program is incorrect" );
      ensure( de != NULL, "Directory has been corrupted" );

      ce->finishTransfer( cd.dirVersion, OUTPUT, true );

      // Make sure we are synchronizing the newest version
      if ( de->getOwner() == sd._this && ce->getVersion() == cd.getDirectoryVersion()) {
          de->clearOwnerCS( sd._this );
      }
   }

   if ( cd.copying ) {
      ce->finishTransfer( cd.dirVersion, INPUT, false );
   }
}

template <class _T>
inline void DeviceCache<_T>::synchronize( CopyDescriptor& cd )
{
   SyncData sd = { this };
   synchronizeInternal( sd, cd );
}

template <class _T>
inline void DeviceCache<_T>::synchronize( std::list<CopyDescriptor> &cds )
{
   SyncData sd = { this };
   for ( std::list<CopyDescriptor>::iterator it = cds.begin(); it != cds.end(); it++ ) {
      synchronizeInternal( sd, *it );
   }
//   FIXME: Does for_each only work with basic types or am I missing some method in the CopyDescriptor?
//   for_each( cds.begin(), cds.end(), std :: bind1st( std :: ptr_fun ( synchronizeInternal ), sd ) );
}

template <class _T>
inline void DeviceCache<_T>::waitInput( uint64_t tag )
{
   CacheEntry *ce = _cache.find(tag);
   ensure( ce != NULL, "Cache has been corrupted" );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_WAIT_INPUT ); )
   while ( ce->isCopying() ) {}
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )

}

template <class _T>
inline void DeviceCache<_T>::waitInput( DeviceCache<_T>* _this, uint64_t tag )
{
   CacheEntry *ce = _this->_cache.find(tag);
   ensure( ce != NULL, "Cache has been corrupted" );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ), NANOS_CACHE_EVENT_WAIT_INPUT ); )
   while ( ce->isCopying() ) {}
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "cache-wait" ) ); )

}

template <class _T>
inline void DeviceCache<_T>::waitInputs( std::list<uint64_t> &tags )
{
   for_each( tags.begin(), tags.end(), std :: bind1st( std :: ptr_fun ( waitInput ), this ) );
   for_each( tags.begin(), tags.end(), waitInput );
}

template <class _T>
inline void DeviceCache<_T>::invalidate( Directory &dir, uint64_t tag, DirectoryEntry *de )
{
  de->trySetInvalidated();
}

template <class _T>
inline void DeviceCache<_T>::invalidateAndFlush( Directory &dir, uint64_t tag, DirectoryEntry *de )
{
   CacheEntry *ce = _cache.find( tag );

   if ( de->trySetInvalidated() ) {
      if ( de->getOwner() != this ) {
         // someone flushed it between setting to invalidated and setting to flushing, do nothing
      } else {
         CopyDescriptor cd = CopyDescriptor( tag, de->getVersion(), /* copying */ false, /* flushing */ true );
         ce->addTransfer( cd.dirVersion, OUTPUT, true );
         if ( copyBackFromCache( cd, ce->getSize() ) ) {
            ce->finishTransfer( cd.dirVersion, OUTPUT, true );
            cd.flushing = false;
            if ( !ce->isFlushing() ) {
               de->setOwner( NULL );
            }
         }
      }
   }
} 

template <class _T>
inline void DeviceCache<_T>::invalidateAndFlush( Directory &dir, uint64_t tag, size_t size, DirectoryEntry *de )
{
   CacheEntry *ce = _cache.find( tag );

   if ( de->trySetInvalidated() ) {
      if ( de->getOwner() != this ) {
         // someone flushed it between setting to invalidated and setting to flushing, do nothing
      } else {
         CopyDescriptor cd = CopyDescriptor( tag, de->getVersion(), /* copying */ false, /* flushing */ true );
         ce->addTransfer( cd.dirVersion, OUTPUT, true );
         if ( copyBackFromCache( cd, size ) ) {
            ce->finishTransfer( cd.dirVersion, OUTPUT, true );
            cd.flushing = false;
            if ( !ce->isFlushing() ) {
               de->setOwner( NULL );
            }
         }
      }
   }
}

template <class _T>
inline size_t& DeviceCache<_T>::getCacheSize()
{
   return _size;
}

template <class _T>
inline void DeviceCache<_T>::syncTransfer( uint64_t tag )
{
   _T::syncTransfer( tag, _pe );
}

template <class _T>
int DeviceCache<_T>::getReferences( unsigned int tag )
{
   return _cache.find( tag )->getReferences();
}

template <class _T>
void DeviceCache<_T>::getUnreferencedEntries(std::list<CacheEntry *> &entries)
{
   for ( CacheHash::iterator it = _cache.begin(); it != _cache.end(); it++ ) {
      CacheEntry &ce = *it;
      if ( ce.getReferences() == 0 ) entries.push_back( &ce );
   }
}

#endif
