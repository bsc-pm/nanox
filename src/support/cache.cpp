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

#include "cache.hpp"

namespace nanos {
void AsynchronousWriteThroughPolicy::registerCacheAccess( uint64_t tag, size_t size, bool input, bool output )
{
   _directory.lock();
   DirectoryEntry *de = _directory.getEntry( tag );

   if ( de == NULL ) { // Memory access not registered in the directory
      // Create directory entry, if the access is output, own it
      de = &( _directory.newEntry( tag, 0, ( output ? &_cache : NULL ) ) );

      // Create cache entry (we assume it is not in the cache)
      CacheEntry& ce = _cache.newEntry( tag, 0, output, size );
      ce.increaseRefs();
      ce.setAddress( _cache.allocate( size ) );

      // Need to copy in ?
      if ( input ) {
         _cache.copyDataToCache( tag, size );
      }
   }
   else {
      Cache *owner = de->getOwner();

      if ( owner != NULL ) {
         // FIXME Is dirty we need to interact with the other cache
         CacheEntry *ce = _cache.getEntry( tag );
         if ( ce == NULL ) {
            // I'm not the owner
            CacheEntry& nce = _cache.newEntry( tag, 0, output, size );
            nce.increaseRefs();
            nce.setAddress( _cache.allocate( size ) );

            // Need to copy in ?
            if ( input ) {
               _directory.unLock();
               while ( owner->getEntry( tag )->isDirty() ) {
                  myThread->idle();
               }
               _directory.lock();
               _cache.copyDataToCache( tag, size );
               nce.setVersion( de->getVersion() );
            }

            if ( output ) {
               de->setOwner( &_cache );
               nce.setDirty( true );
               de->setVersion( de->getVersion() + 1 ) ;
               nce.setVersion( de->getVersion() );
            }
         }
         else
         {
            if ( ( ce->getVersion() < de->getVersion() ) && input ) {
               _directory.unLock();
               if ( owner != &_cache ) {
                  // Check I'm not the owner
                  while ( owner->getEntry( tag )->isDirty() ) {
                     myThread->idle();
                  }
               }
               _directory.lock();
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
      }
      else {
         // lookup in cache
         CacheEntry *ce = _cache.getEntry( tag );
         if ( ce != NULL ) {
            if ( ( ce->getVersion() < de->getVersion() ) && input ) {
               _cache.copyDataToCache( tag, size );
               ce->setVersion( de->getVersion() );
            }
            else {
               // Hit in the cache
               NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-hit") );
               NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) tag ) );
            }

            if ( output ) {
               de->setOwner( &_cache );
               ce->setDirty( true );
               de->setVersion( de->getVersion() + 1 ) ;
               ce->setVersion( de->getVersion() );
            }
         }
         else {
            if ( output ) {
               de->setOwner( &_cache );
               de->setVersion( de->getVersion() + 1 );
            }
            ce = & (_cache.newEntry( tag, de->getVersion(), output, size ) );
            //ce->increaseRefs();
            ce->setAddress( _cache.allocate( size ) );
            if ( input ) {
               _cache.copyDataToCache( tag, size );
            }
         }
         ce->increaseRefs();
      }
   }
   _directory.unLock();
}


void AsynchronousWriteThroughPolicy::unregisterCacheAccess( uint64_t tag, size_t size )
{
   _directory.lock();
   CacheEntry *ce = _cache.getEntry( tag );
   //ensure (ce != NULL, "Cache has been corrupted");
   if ( ce->isDirty() ) {
      _cache.copyBackFromCache( tag, size );
   }
   _directory.unLock();
}

void AsynchronousWriteThroughPolicy::updateCacheAccess( uint64_t tag, size_t size )
{
   _directory.lock();
   CacheEntry *ce = _cache.getEntry( tag );
   DirectoryEntry *de = _directory.getEntry( tag );
   de->setOwner( NULL );
   ce->setDirty( false );
   ce->decreaseRefs();
   _directory.unLock();
}

void AsynchronousWriteThroughPolicy::registerPrivateAccess( uint64_t tag, size_t size, bool input, bool output )
{
   // Private accesses are never found in the cache, the directory is not used because they can't be shared
   CacheEntry& ce = _cache.newEntry( tag, 0, output, size );
   ce.increaseRefs();
   ce.setAddress( _cache.allocate( size ) );
   if ( input )
      _cache.copyDataToCache( tag, size );
}

void AsynchronousWriteThroughPolicy::unregisterPrivateAccess( uint64_t tag, size_t size )
{
   CacheEntry *ce = _cache.getEntry( tag );
   ensure ( ce != NULL, "Private access cannot miss in the cache.");
   if ( ce->isDirty() )
      _cache.copyBackFromCache( tag, size );
   _cache.deleteEntry( tag, size );
}


}
