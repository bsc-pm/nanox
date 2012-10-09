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
#include "regioncache.hpp"
#include "system.hpp"

using namespace nanos;


//template <class CacheDevice>
//void CachedAccelerator<CacheDevice>::configureCache( std::size_t cacheSize, System::CachePolicyType cachePolicy )
//{
//   if ( _cache == NULL )
//      _cache = NEW DeviceCache<CacheDevice>( cacheSize, NULL, this );
//
//   switch ( cachePolicy ) {
//      case System::NONE:
//         _cachePolicy = NEW NoCache( *_cache );
//         break;
//      case System::WRITE_THROUGH:
//         _cachePolicy = NEW WriteThroughPolicy( *_cache );
//         break;
//      case System::WRITE_BACK:
//         _cachePolicy = NEW WriteBackPolicy( *_cache );
//         break;
//      default:
//         // We should not get here with the System::DEFAULT value
//         fatal0( "Unknown cache policy" );
//         break;
//   }
//
//   _cache->setPolicy( _cachePolicy );
//}

#if 0
template <class CacheDevice>
inline void CachedAccelerator<CacheDevice>::registerCacheAccessDependent( Directory& dir, uint64_t tag, size_t size, bool input, bool output )
{
   _cache->registerCacheAccess( dir, tag, size, input, output );
}

template <class CacheDevice>
inline void CachedAccelerator<CacheDevice>::unregisterCacheAccessDependent( Directory& dir, uint64_t tag, size_t size, bool output )

{
   _cache->unregisterCacheAccess( dir, tag, size, output );
}

template <class CacheDevice>
inline void CachedAccelerator<CacheDevice>::registerPrivateAccessDependent( Directory& dir, uint64_t tag, size_t size, bool input, bool output )
{
   _cache->registerPrivateAccess( dir, tag, size, input, output );
}

template <class CacheDevice>
inline void CachedAccelerator<CacheDevice>::unregisterPrivateAccessDependent( Directory& dir, uint64_t tag, size_t size )
{
   _cache->unregisterPrivateAccess( dir, tag, size );
}

template <class CacheDevice>
inline void CachedAccelerator<CacheDevice>::synchronize( CopyDescriptor &cd )
{
   _cache->synchronize( cd );
}

template <class CacheDevice>
inline void CachedAccelerator<CacheDevice>::synchronize( std::list<CopyDescriptor> &cds )
{
   _cache->synchronize( cds );
}

template <class CacheDevice>
inline void CachedAccelerator<CacheDevice>::waitInputDependent( uint64_t tag )
{
   _cache->waitInput( tag );
}

template <class CacheDevice>
inline void* CachedAccelerator<CacheDevice>::getAddressDependent( uint64_t tag )
{
   return _cache->getAddress( tag );
}

template <class CacheDevice>
inline void* CachedAccelerator<CacheDevice>::newGetAddressDependent( CopyData const &cd )
{
   uint64_t addr = 0xdeabeef; //d_newCache.getAddress( cd );
   std::cerr << "new cache says: " << (void *) addr;
   return (void *) addr;
}

template <class CacheDevice>
inline void CachedAccelerator<CacheDevice>::copyToDependent( void *dst, uint64_t tag, size_t size )
{
   _cache->copyTo( dst, tag, size );
}

template <class CacheDevice>
inline bool CachedAccelerator<CacheDevice>::checkBlockingCacheAccessDependent( Directory &dir, uint64_t tag, size_t size, bool input, bool output )
{
   return _cache->checkBlockingCacheAccess( dir, tag, size, input, output ) ;
}
#endif
inline void CachedAccelerator::copyDataInDependent( WorkDescriptor &wd )
{
   //wd._ccontrol.preInit( wd.getNewDirectory(), wd.getNumCopies(), wd.getCopies(), wd.getId() );
   wd._ccontrol.copyDataIn( &_newCache );
}
inline void CachedAccelerator::waitInputsDependent( WorkDescriptor &wd )
{
   //std::cerr << "waiting for inputs... wd " << wd.getId() << std::endl;
   while ( !wd._ccontrol.dataIsReady() ) { myThread->idle(); } 
   //std::cerr << "waiting for inputs done wd " << wd.getId() << std::endl;
}

inline Device const *CachedAccelerator::getCacheDeviceType( ) const {
   return &_newCache.getDevice();
}

inline RegionCache *CachedAccelerator::getCache() {
   return &_newCache;
}

#endif
