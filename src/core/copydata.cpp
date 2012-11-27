#include "copydata.hpp"
#include <ostream>

using namespace nanos;

std::size_t CopyData::getWideSizeRecursive( int i ) const
{
   if ( i == 0 )
      return dimensions[ 0 ].size;
   else
      return dimensions[ i ].size * getWideSizeRecursive( i - 1 );
}

std::size_t CopyData::getFitSizeRecursive( int i ) const
{
   if ( i == 0 )
      return dimensions[ 0 ].accessed_length;
   else if ( dimensions[ i ].accessed_length == 1 ) 
      return getFitSizeRecursive( i - 1 );
   else
      return dimensions[ i ].accessed_length * getWideSizeRecursive( i - 1 );
}

uint64_t CopyData::getFitOffsetRecursive( int i ) const
{
   uint64_t off = dimensions[ i ].lower_bound * ( i == 0 ? 1 : getWideSizeRecursive( i - 1 ) );
   if ( dimensions[ i ].accessed_length == 1 )
      off += getFitOffsetRecursive( i - 1 );
   return off;
}

std::ostream& nanos::operator<< (std::ostream &o, CopyData const &cd) {
   o << "CopyData" << std::endl;
   o << "\tAddtess: " << cd.address << std::endl;
   o << "\tOffset: " << cd.offset << std::endl;
   o << "\tDimensions: " << cd.dimension_count << std::endl;
   for ( int i = 0; i < cd.dimension_count; i++) {
      o << "\t\t[" << i << "] size: " << cd.dimensions[i].size << ", accessed length: " << cd.dimensions[i].accessed_length << ", lower bound: " << cd.dimensions[i].lower_bound << std::endl;
   } 
   o << "End of CopyData" << std::endl;
   return o;
}

