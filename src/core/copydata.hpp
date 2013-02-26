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

#ifndef _NANOS_COPYDATA
#define _NANOS_COPYDATA

#include "copydata_decl.hpp"

using namespace nanos;

inline CopyData::CopyData ( uint64_t addr, nanos_sharing_t nxSharing, bool input, bool output, std::size_t numDimensions, nanos_region_dimension_internal_t const *dims, ptrdiff_t off )
{
   address = (void *) addr;
   sharing = nxSharing;
   flags.input = input;
   flags.output = output;
   dimension_count = numDimensions;
   dimensions = dims;
   offset = off;
}

inline CopyData::CopyData ( const CopyData &cd )
{
   address = cd.address;
   sharing = cd.sharing;
   flags.input = cd.flags.input;
   flags.output = cd.flags.output;
   dimension_count = cd.dimension_count;
   dimensions = cd.dimensions;
   offset = cd.offset;
}

inline const CopyData & CopyData::operator= ( const CopyData &cd )
{
   if ( this == &cd ) return *this; 
   address = cd.address;
   sharing = cd.sharing;
   flags.input = cd.flags.input;
   flags.output = cd.flags.output;
   dimension_count = cd.dimension_count;
   dimensions = cd.dimensions;
   offset = cd.offset;
   return *this;
}

inline void *CopyData::getBaseAddress() const
{
   return address;
}

inline void CopyData::setBaseAddress( void *addr )
{
   address = addr;
}

inline bool CopyData::isInput() const
{
   return flags.input;
}

inline void CopyData::setInput( bool b )
{
   flags.input = b;
}

inline bool CopyData::isOutput() const
{
   return flags.output;
}

inline void CopyData::setOutput( bool b )
{
   flags.output = b;
}

inline std::size_t CopyData::getSize() const
{
   std::size_t size = 0;
   ensure( dimension_count >= 1, "Wrong dimension_count ");
   size = dimensions[0].accessed_length;
   for ( int i = 1; i < dimension_count; i += 1 )
      size *= dimensions[i].accessed_length;
   return size;
}

inline bool CopyData::isShared() const
{
   return sharing ==  NANOS_SHARED;
}

inline bool CopyData::isPrivate() const
{
   return sharing ==  NANOS_PRIVATE;
}

inline nanos_sharing_t CopyData::getSharing() const
{
   return sharing;
}

inline std::size_t CopyData::getNumDimensions() const
{
   return dimension_count;
}

inline nanos_region_dimension_internal_t const *CopyData::getDimensions() const
{
   return dimensions;
}

inline void CopyData::setDimensions(nanos_region_dimension_internal_t const *dims)
{
   dimensions = dims;
}

inline uint64_t CopyData::getAddress() const
{
   return ( (uint64_t) address ); 
}
#endif
