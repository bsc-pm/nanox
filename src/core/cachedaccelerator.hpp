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

#ifndef _NANOS_CACHED_ACCELERATOR
#define _NANOS_CACHED_ACCELERATOR

#include "cachedaccelerator_decl.hpp"
#include "accelerator_decl.hpp"
#include "cache.hpp"

using namespace nanos;


template <class Device>
void CachedAccelerator<Device>::configureCache( size_t cacheSize, System::CachePolicyType cachePolicy )
{
   if ( _cache == NULL )
      _cache = NEW DeviceCache<Device>( cacheSize, NULL, this );

   switch ( cachePolicy ) {
      case System::NONE:
         _cachePolicy = NEW NoCache( *_cache );
         break;
      case System::WRITE_THROUGH:
         _cachePolicy = NEW WriteThroughPolicy( *_cache );
         break;
      case System::WRITE_BACK:
         _cachePolicy = NEW WriteBackPolicy( *_cache );
         break;
      default:
         // We should not get here with the System::DEFAULT value
         fatal0( "Unknown cache policy" );
         break;
   }

   _cache->setPolicy( _cachePolicy );
}

template <class Device>
inline void CachedAccelerator<Device>::registerCacheAccessDependent( Directory& dir, uint64_t tag, size_t size, bool input, bool output )
{
   _cache->registerCacheAccess( dir, tag, size, input, output );
}

template <class Device>
inline void CachedAccelerator<Device>::unregisterCacheAccessDependent( Directory& dir, uint64_t tag, size_t size, bool output )
{
   _cache->unregisterCacheAccess( dir, tag, size, output );
}

template <class Device>
inline void CachedAccelerator<Device>::registerPrivateAccessDependent( Directory& dir, uint64_t tag, size_t size, bool input, bool output )
{
   _cache->registerPrivateAccess( dir, tag, size, input, output );
}

template <class Device>
inline void CachedAccelerator<Device>::unregisterPrivateAccessDependent( Directory& dir, uint64_t tag, size_t size )
{
   _cache->unregisterPrivateAccess( dir, tag, size );
}

template <class Device>
inline void CachedAccelerator<Device>::synchronize( CopyDescriptor &cd )
{
   _cache->synchronize( cd );
}

template <class Device>
inline void CachedAccelerator<Device>::synchronize( std::list<CopyDescriptor> &cds )
{
   _cache->synchronize( cds );
}

template <class Device>
inline void CachedAccelerator<Device>::waitInputDependent( uint64_t tag )
{
   _cache->waitInput( tag );
}

template <class Device>
inline void* CachedAccelerator<Device>::getAddressDependent( uint64_t tag )
{
   return _cache->getAddress( tag );
}

template <class Device>
inline void CachedAccelerator<Device>::copyToDependent( void *dst, uint64_t tag, size_t size )
{
   _cache->copyTo( dst, tag, size );
}

#endif
