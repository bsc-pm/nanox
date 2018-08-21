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

#ifndef _NANOS_COPYDATA
#define _NANOS_COPYDATA

#include "copydata_decl.hpp"
//#include "system_decl.hpp"

namespace nanos {

inline CopyData::CopyData ( uint64_t addr, nanos_sharing_t nxSharing, bool input, bool output, std::size_t numDimensions, nanos_region_dimension_internal_t const *dims, ptrdiff_t off, uint64_t hostBaseAddress, reg_t hostRegionId )
{
   address = (void *) addr;
   sharing = nxSharing;
   flags.input = input;
   flags.output = output;
   dimension_count = numDimensions;
   dimensions = dims;
   offset = off;
   host_base_address = hostBaseAddress;
   host_region_id = hostRegionId;
   remote_host = false;
   deducted_cd = NULL;
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
   host_base_address = cd.host_base_address;
   host_region_id = cd.host_region_id;
   remote_host = cd.remote_host;
   deducted_cd = cd.deducted_cd;
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
   host_base_address = cd.host_base_address;
   host_region_id = cd.host_region_id;
   remote_host = cd.remote_host;
   deducted_cd = cd.deducted_cd;
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
   //ensure( dimension_count >= 1, "Wrong dimension_count ");
   size = dimensions[0].accessed_length;
   for ( int i = 1; i < dimension_count; i += 1 )
      size *= dimensions[i].accessed_length;
   return size;
}
inline std::size_t CopyData::getMaxSize() const
{
   std::size_t size = 0;
   //ensure( dimension_count >= 1, "Wrong dimension_count ");
   size = dimensions[0].size;
   for ( int i = 1; i < dimension_count; i += 1 )
      size *= dimensions[i].size;
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

inline void CopyData::setNumDimensions(std::size_t ndims) {
   dimension_count = ndims;
}

inline nanos_region_dimension_internal_t const *CopyData::getDimensions() const
{
   return dimensions;
}

inline void CopyData::setDimensions(nanos_region_dimension_internal_t *dims)
{
   dimensions = dims;
}

inline uint64_t CopyData::getAddress() const
{
   return ( (uint64_t) address ); 
}

inline uint64_t CopyData::getOffset() const
{
   return (uint64_t) offset; 
}

inline std::size_t CopyData::getFitSize() const
{
   return getFitSizeRecursive( dimension_count - 1 );
}

inline uint64_t CopyData::getFitAddress() const
{
   return ( (uint64_t) getBaseAddress() ) + getFitOffsetRecursive( dimension_count - 1 );
}

inline uint64_t CopyData::getHostBaseAddress() const {
   return host_base_address;
}

inline void CopyData::setHostBaseAddress( uint64_t addr ) {
   host_base_address = addr;
}

inline reg_t CopyData::getHostRegionId() const {
   return host_region_id;
}

inline void CopyData::setHostRegionId( reg_t id ) {
   host_region_id = id;
}

inline bool CopyData::isRemoteHost() const {
   return remote_host;
}

inline void CopyData::setRemoteHost( bool value ) {
   remote_host = value;
}

inline void CopyData::setDeductedCD( CopyData *cd ) {
   deducted_cd = (void *) cd;
}

inline CopyData *CopyData::getDeductedCD() {
   return (CopyData *) deducted_cd;
}

} // namespace nanos

#endif
