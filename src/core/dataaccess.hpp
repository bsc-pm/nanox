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

#ifndef _NANOS_DATA_ACCESS
#define _NANOS_DATA_ACCESS

#include "dataaccess_decl.hpp"
#include <iostream>

namespace nanos {

inline DataAccess::DataAccess ( void * addr, bool input, bool output,
             bool canRenameFlag, bool concurrent,bool commutative,
             short dimensionCount, nanos_region_dimension_internal_t const *dims,
             ptrdiff_t someOffset )
{
   address = addr;
   flags.input = input;
   flags.output = output;
   flags.can_rename = canRenameFlag;
   flags.concurrent = concurrent; 
   flags.commutative = commutative;
   dimension_count = dimensionCount;
   dimensions = dims;
   offset = someOffset;
}

inline DataAccess::DataAccess ( const DataAccess &dataAccess )
{
   address = dataAccess.address;
   flags.input = dataAccess.flags.input;
   flags.output = dataAccess.flags.output;
   flags.can_rename = dataAccess.flags.can_rename;
   flags.concurrent = dataAccess.flags.concurrent; 
   flags.commutative = dataAccess.flags.commutative;
   dimension_count = dataAccess.dimension_count;
   dimensions = dataAccess.dimensions;
   offset = dataAccess.offset;
}

inline const DataAccess & DataAccess::operator= ( const DataAccess &dataAccess )
{
   if ( this == &dataAccess ) return *this; 
   address = dataAccess.address;
   flags.input = dataAccess.flags.input;
   flags.output = dataAccess.flags.output;
   flags.can_rename = dataAccess.flags.can_rename;
   flags.concurrent = dataAccess.flags.concurrent; 
   flags.commutative = dataAccess.flags.commutative;
   dimension_count = dataAccess.dimension_count;
   dimensions = dataAccess.dimensions;
   offset = dataAccess.offset;
   return *this;
}

inline void * DataAccess::getAddress() const
{
   return address;
}

inline void * DataAccess::getDepAddress() const
{
   return (void*)((uintptr_t)address + offset );
}

inline ptrdiff_t DataAccess::getOffset() const
{
   return offset;
}
inline bool DataAccess::isInput() const
{
   return flags.input;
}

inline void DataAccess::setInput( bool b )
{
 flags.input = b;
}

inline bool DataAccess::isOutput() const
{
   return flags.output;
}

inline void DataAccess::setOutput( bool b )
{
   flags.output = b;
}

inline bool DataAccess::canRename() const
{
   return flags.can_rename;
}

inline void DataAccess::setCanRename( bool b )
{
   flags.can_rename = b;
}

inline bool DataAccess::isConcurrent() const
{ 
   return flags.concurrent;
}
 
inline void DataAccess::setConcurrent( bool b )
{ 
   flags.concurrent = b;
}

inline bool DataAccess::isCommutative() const
{
   return flags.commutative;
}

inline void DataAccess::setCommutative( bool b )
{
   flags.commutative = b;
}

inline std::size_t DataAccess::getSize() const
{
   std::size_t size = dimensions[dimension_count-1].accessed_length;
   for ( int i = 0; i < dimension_count-1 ; i++ )
      size *= dimensions[i].size;
   return size;
}

               
/*! \brief gets the pointer of the dimensions
  */
inline nanos_region_dimension_internal_t const* DataAccess::getDimensions() const {
    return dimensions;
}

/*! \brief gets the number of dimensions
  */
inline short DataAccess::getNumDimensions() const {
    return dimension_count;
}


namespace dependencies_domain_internal {
   inline std::ostream & operator<<( std::ostream &o, AccessType const &accessType)
   {
      if ( accessType.input && accessType.output ) {
         if ( accessType.concurrent ) {
            o << "CON";
         } else if ( accessType.commutative ) {
            o << "COM";
         } else {
            o << "INOUT";
         }
      } else if ( accessType.input && !accessType.commutative ) {
         o << "IN";
      } else if ( accessType.output && !accessType.commutative ) {
         o << "OUT";
      } else {
         o << "ERR";
      }
      return o;
   }
}

} // namespace nanos

#endif
