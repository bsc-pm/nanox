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

#ifndef _NANOS_DEPSREGION_H
#define _NANOS_DEPSREGION_H

#include "depsregion_decl.hpp"

namespace nanos {

inline const DepsRegion& DepsRegion::operator= ( const DepsRegion &obj )
{
   _address = obj._address;
   _endAddress = obj._endAddress; 
   _trackable = obj._trackable;
   return *this;
}

inline const DepsRegion::TargetType& DepsRegion::operator() () const
{
   return _address;
}

inline bool DepsRegion::operator== ( const DepsRegion &obj ) const
{       
    return _address==obj._address && _endAddress == obj._endAddress;
}

inline bool DepsRegion::overlap ( const BaseDependency &obj ) const
{       
   const DepsRegion& region( static_cast<const DepsRegion&>( obj ) );   
   return !( region._endAddress<_address || region._address>_endAddress );
}
 
inline bool DepsRegion::operator< ( const DepsRegion &obj ) const
{
   return _address < obj._address;
}

inline BaseDependency* DepsRegion::clone() const
{
   return new DepsRegion( _address, _endAddress, _trackable );
}

inline void * DepsRegion::getAddress () const
{
   return _address;
}

inline void * DepsRegion::getEndAddress () const
{
   return _endAddress;
}


inline size_t DepsRegion::getSize () const
{
   return (uint64_t)_endAddress-(uint64_t)_address;
}

} // namespace nanos

#endif
