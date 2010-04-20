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

  /*! \brief A Cache is a class that provides basic services for registering and
   *         searching for memory blocks in a device using an identifier represented
   *         by an unsigned int of 64 bits which represents the address of the original
   *         data in the host. 
   */
   template <class _T>
   class Cache
   {
     /* FIXME (see #195)
      *   - Code in the cache c file?
      *   - check for errors 
      */
      private:
        /*! \brief Represents a cache entry identified by an address
         */
         class CacheEntry
         {
            private:
               /**< Address identifier of the cache entry  */
               void *_addr;

               /**< Entry references counter  */
               unsigned int _refs;

            public:
              /*! \brief Default constructor
               */
               CacheEntry(): _addr( NULL), _refs(0) {}

              /*! \brief Constructor
               *  \param addr: address of the cache entry
               */
               CacheEntry( void *addr ): _addr( addr ), _refs(0) {}

              /*! \brief Copy constructor
               *  \param Another CacheEntry
               */
               CacheEntry( const CacheEntry &ce ): _addr( ce._addr ), _refs( ce._refs ) {}

              /* \brief Destructor
               */
               ~CacheEntry() {}

              /* \brief Assign operator
               */
               CacheEntry& operator=( const CacheEntry &ce )
               {
                  if ( this == &ce ) return *this;
                  this->_addr = ce._addr;
                  this->_refs = ce._refs;
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
         };

         /**< Maps keys with CacheEntries  */
         typedef TR1::unordered_map< uint64_t, CacheEntry> CacheHash;
         CacheHash _cache;

         // disable copy constructor and assignment operator
         Cache( const Cache &cache );
         const Cache & operator= ( const Cache &cache );

      public:
        /* \brief Default constructor
         */
         Cache() : _cache() {}

        /* \brief Register a new entry in the cache
         * \param tag: identifier of the entry
         * \param size: size to allocate if the entry was not in the cache
         */
         void cacheData( uint64_t tag, size_t size )
         {
            CacheEntry &entry = _cache[tag];
            if ( entry.hasRefs() ) {
               entry.increaseRefs();
            } else {
               entry.setAddress( _T::allocate( size ) );
               entry.increaseRefs();
            }
         }

        /* \brief Deleting a reference from a given entry
         * \param tag: identifier of the entry
         */
         void flush( uint64_t tag )
         {
            CacheEntry &entry = _cache[tag];
            entry.decreaseRefs();
         }

        /* \brief Copy data from the address represented by the tag to the entry in the device.
         * \param tag: identifier of the entry
         * \param size: number of bytes to copy
         */
         void copyData( uint64_t tag, size_t size )
         {
            _T::copyIn( _cache[tag].getAddress(), tag, size );
         }

        /* \brief Copy back from the entry to the address represented by the tag.
         * \param tag: Entry identifier and address of original data
         * \param size: number of bytes to copy
         */
         void copyBack( uint64_t tag, size_t size )
         {
            CacheEntry &entry = _cache[tag];
            _T::copyOut( tag, entry.getAddress(), size );
            if ( !entry.hasRefs() ) {
               _T::free( entry.getAddress() );
               entry.setAddress(NULL);
            }
         }

        /* \brief get the Address in the cache for tag
         * \param tag: Identifier of the entry to look for
         */
         void * getAddress( uint64_t tag )
         {
            void *result = _cache[tag].getAddress();
            return result;
         }

        /* \brief Perform local copy in the device for an entry
         * \param dst: Device destination address to copy to
         * \param tag: entry identifier to look for the source data
         * \param size: number of bytes to copy
         */
         void copyTo( void *dst, uint64_t tag, size_t size )
         {
            _T::copyLocal( dst, _cache[tag].getAddress(), size );
         }
   };
}

#endif
