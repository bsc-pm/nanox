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

#ifndef _NANOS_DATA_ACCESS
#define _NANOS_DATA_ACCESS

#include "dataaccess_decl.hpp"

using namespace nanos;

inline DataAccess::DataAccess ( void * addr, bool input, bool output,
             bool canRenameFlag, bool commutative, short dimensionCount,
             nanos_region_dimension_internal_t const *dims )
{
   address = addr;
   flags.input = input;
   flags.output = output;
   flags.can_rename = canRenameFlag;
   flags.commutative = commutative;
   dimension_count = dimensionCount;
   dimensions = dims;
}

inline DataAccess::DataAccess ( const DataAccess &dataAccess )
{
   address = dataAccess.address;
   flags.input = dataAccess.flags.input;
   flags.output = dataAccess.flags.output;
   flags.can_rename = dataAccess.flags.can_rename;
   flags.commutative = dataAccess.flags.commutative;
   dimension_count = dataAccess.dimension_count;
   dimensions = dataAccess.dimensions;
}

inline const DataAccess & DataAccess::operator= ( const DataAccess &dataAccess )
{
   if ( this == &dataAccess ) return *this; 
   address = dataAccess.address;
   flags.input = dataAccess.flags.input;
   flags.output = dataAccess.flags.output;
   flags.can_rename = dataAccess.flags.can_rename;
   flags.commutative = dataAccess.flags.commutative;
   dimension_count = dataAccess.dimension_count;
   dimensions = dataAccess.dimensions;
   return *this;
}

inline void * DataAccess::getAddress() const
{
   return address;
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

inline bool DataAccess::isCommutative() const
{
   return flags.commutative;
}

inline void DataAccess::setCommutative( bool b )
{
   flags.commutative = b;
}

#endif
