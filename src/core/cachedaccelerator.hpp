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

inline CachedAccelerator::CachedAccelerator( int newId, const Device *arch,
   const Device *subArch, Device *cacheArch, std::size_t cacheSize, enum RegionCache::CacheOptions flags, memory_space_id_t addressSpace ) :
   Accelerator( newId, arch, subArch ), _addressSpaceId( addressSpace ) , _newCache( (memory_space_id_t ) -1, *cacheArch, flags )  {
   //sys.getCaches()[this->getMemorySpaceId()] = &_newCache;
}

inline CachedAccelerator::~CachedAccelerator() {
}

inline void CachedAccelerator::copyDataInDependent( WorkDescriptor &wd )
{
   //wd._ccontrol.copyDataIn( &_newCache );
   wd._mcontrol.copyDataIn( *this );
}

inline void CachedAccelerator::waitInputsDependent( WorkDescriptor &wd )
{
   while ( !wd._mcontrol.isDataReady() ) { myThread->idle(); } 
   //while ( !wd._ccontrol.dataIsReady() ) { myThread->idle(); } 
}

inline Device const *CachedAccelerator::getCacheDeviceType( ) const {
   return &_newCache.getDevice();
}

#endif
