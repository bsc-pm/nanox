/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#ifndef _NANOS_CACHE_MAP_HPP
#define _NANOS_CACHE_MAP_HPP
#include "cache_map_decl.hpp"
#include "new_decl.hpp"

namespace nanos {

inline unsigned int CacheMap::registerCache()
{
   return _numCaches++;
}

inline unsigned int CacheMap::getSize() const
{
   return _numCaches.value() - 1;
}

inline CacheAccessMap::CacheAccessMap( unsigned int size ) : _size(size)
{
   //ensure( size > 0, "Can not create a CacheAccessMap with size=0.")
   if ( size > 0 ) /* this if statement should be removed, we should assume size is never <= 0 */
   {
      _cacheAccessesById = NEW Atomic<unsigned int>[size];
      for ( unsigned int i = 0; i < size; i += 1 )
         _cacheAccessesById[ i ] = 0;
   }
   else
   {
      _cacheAccessesById = NULL;
   }
}

inline CacheAccessMap::~CacheAccessMap()
{
   delete[] _cacheAccessesById;
}

inline CacheAccessMap::CacheAccessMap( const CacheAccessMap &map ) : _size( map._size )
{
   if ( this == &map )
      return;
   //ensure( _size > 0, "Can not create a CacheAccessMap with size=0.")
   if ( _size > 0 )
   {
      _cacheAccessesById = NEW Atomic<unsigned int>[map._size];
      for ( unsigned int i = 0; i < map._size; i++ ) {
         _cacheAccessesById[i] = map._cacheAccessesById[i];
      }
   } else {
      _cacheAccessesById = NULL;
   }
}

inline const CacheAccessMap& CacheAccessMap::operator= ( const CacheAccessMap &map )
{
   if ( this == &map )
      return *this;
   _size = map._size;
   //ensure( _size > 0, "Can not create a CacheAccessMap with size=0.")
   if ( _size > 0 )
   {
      _cacheAccessesById = NEW Atomic<unsigned int>[_size];
      for ( unsigned int i = 0; i < _size; i++ ) {
         _cacheAccessesById[i] = map._cacheAccessesById[i];
      }
   } else {
      _cacheAccessesById = NULL;
   }
   return *this;
}

inline Atomic<unsigned int>& CacheAccessMap::operator[] ( unsigned int cacheId )
{
   ensure( cacheId != 0, "Checking an invalid CacheAccessMap id = 0")
   ensure( cacheId <= _size, "Checking an invalid CacheAccessMap id > size.")
   return _cacheAccessesById[cacheId - 1];
}

inline unsigned int CacheAccessMap::getAccesses( unsigned int cacheId )
{
   ensure( cacheId != 0, "Checking an invalid CacheAccessMap id = 0")
   ensure( cacheId <= _size, "Checking an invalid CacheAccessMap id > size.")
   return _cacheAccessesById[cacheId - 1].value();
}

} // namespace nanos

#endif
