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

#ifndef MEMCACHECOPY_DECL
#define MEMCACHECOPY_DECL
#include "memoryops_decl.hpp"
#include "regioncache_decl.hpp"
#include "invalidationcontroller_decl.hpp"
#include <fstream>
namespace nanos {
   class MemCacheCopy {
         unsigned int         _version;
         unsigned int         _childrenProducedVersion;
      public:
         global_reg_t         _reg;
         NewLocationInfoList  _locations;
         bool                 _locationDataReady;
         AllocatedChunk      *_chunk;
         enum RegionCache::CachePolicy _policy;
         InvalidationController _invalControl;
         int                    _allocFrom;
         std::set<reg_t>        _regionsToCommit;
         MemCacheCopy();
         MemCacheCopy( WD const &wd, unsigned int index );

         void getVersionInfo();

         void generateInOps( BaseAddressSpaceInOps &ops, bool input, bool output, WD const &wd, unsigned int copyIdx );
         void generateOutOps( SeparateMemoryAddressSpace *from, SeparateAddressSpaceOutOps &ops, bool input, bool output, WD const &wd, unsigned int copyIdx );
         unsigned int getVersion() const;
         void setVersion( unsigned int version );
         bool isRooted( memory_space_id_t &loc ) const;
         void printLocations( std::ostream &o) const;
         unsigned int getChildrenProducedVersion() const;
         void setChildrenProducedVersion( unsigned int version );
   };
} // namespace nanos
#endif /* MEMCACHECOPY_DECL */
