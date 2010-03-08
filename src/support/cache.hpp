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

namespace nanos {

   template <class _T>
   class Cache
   {
     /* FIXME
      *   - Code in the cache c file?
      *   - check for errors 
      */
      private:
         class CacheEntry
         {
            private:
               void *_addr;
               unsigned int _refs;
            public:
               CacheEntry(): _addr( NULL), _refs(0) {}
               CacheEntry( void *addr ): _addr( addr ), _refs(0) {}
               CacheEntry( const CacheEntry &ce ): _addr( ce._addr ), _refs( ce._refs ) {}

               ~CacheEntry() {}

               CacheEntry& operator=( const CacheEntry &ce )
               {
                  if ( this == &ce ) return *this;
                  this->_addr = ce._addr;
                  this->_refs = ce._refs;
               }

               void * getAddress() const
               { return _addr; }

               void setAddress( void *addr )
               { _addr = addr; _refs++; }

               bool hasRefs() const
               { return _refs > 0; }

               void increaseRefs()
               { _refs++; }

               bool decreaseRefs()
               { return (--_refs) == 0; }
         };

         typedef TR1::unordered_map< uint64_t, CacheEntry> CacheHash;
         CacheHash _cache;
         _T _peMemory;

         // disable copy constructor and assignment operator
         Cache( const Cache &cache );
         const Cache & operator= ( const Cache &cache );

      public:
         Cache() : _cache(), _peMemory() {}

         void cacheData( uint64_t tag, size_t size )
         {
            CacheEntry &entry = _cache[tag];
            if ( entry.hasRefs() ) {
               entry.increaseRefs();
            } else {
               entry.setAddress( _peMemory.allocate( size ) );
            }
         }

         void flush( uint64_t tag )
         {
            CacheEntry &entry = _cache[tag];
            entry.decreaseRefs();
         }

         void copyData( uint64_t tag, size_t size )
         {
            _peMemory.copyIn( _cache[tag].getAddress(), tag, size );
         }

         void copyBack( uint64_t tag, size_t size )
         {
            CacheEntry &entry = _cache[tag];
            _peMemory.copyOut( tag, entry.getAddress(), size );
            if ( !entry.hasRefs() ) {
               _peMemory.free( entry.getAddress() );
               entry.setAddress(NULL);
            }
         }

         void * getAddress( uint64_t tag )
         {
            void *result = _cache[tag].getAddress();
            return result;
         }

         void copyTo( void *dst, uint64_t tag, size_t size )
         {
            _peMemory.copyLocal( dst, _cache[tag].getAddress(), size );
         }
   };
}

#endif
