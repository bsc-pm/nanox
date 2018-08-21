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

#ifndef ADDRESSSPACE_H
#define ADDRESSSPACE_H
#include "addressspace_decl.hpp"
#include "regiondirectory.hpp"

namespace nanos {

template < class T >
void MemSpace< T >::copy( MemSpace< SeparateAddressSpace > &from, TransferList &list, WD const *wd, bool inval ) {
   for ( TransferList::const_iterator it = list.begin(); it != list.end(); it++ ) {
      this->doOp( from, it->getRegion(), it->getVersion(), wd, it->getCopyIndex(), it->getDeviceOps(), it->getDestinationChunk(), it->getSourceChunk(), inval );
   }
}

inline void HostAddressSpace::getVersionInfo( global_reg_t const &reg, unsigned int &version, NewLocationInfoList &locations ) {
   RegionDirectory::__getLocation( reg.key, reg.id, locations, version);
}

inline void HostAddressSpace::getRegionId( CopyData const &cd, global_reg_t &reg, WD const *wd, unsigned int idx ) {
   reg.key = _directory.getRegionDirectoryKeyRegisterIfNeeded( cd, wd );
}

} // namespace nanos

#endif /* ADDRESSSPACE_H */
