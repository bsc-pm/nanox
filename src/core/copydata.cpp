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

#include "copydata.hpp"
#include "basethread.hpp"
#include "debug.hpp"
#include <ostream>

using namespace nanos;

std::size_t CopyData::getWideSizeRecursive( int i ) const
{
   //std::cerr << __FUNCTION__ << " with i " << i << std::endl;
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

void CopyData::getFitDimensions( nanos_region_dimension_internal_t *outDimensions ) const {
   bool keep_fitting = true;
   for ( int idx = dimension_count - 1; idx >= 0; idx -= 1 ) {
      outDimensions[ idx ].size = dimensions[ idx ].size;
      if ( keep_fitting ) {
         outDimensions[ idx ].accessed_length = dimensions[ idx ].accessed_length;
         outDimensions[ idx ].lower_bound = dimensions[ idx ].lower_bound;
         if ( dimensions[ idx ].accessed_length != 1 )
            keep_fitting = false;
      } else {
         outDimensions[ idx ].lower_bound = 0;
         outDimensions[ idx ].accessed_length = dimensions[ idx ].size;
      }
   }
}


void CopyData::deductCd( CopyData const &ref, nanos_region_dimension_internal_t *newDims ) const {
   if ( ref.getNumDimensions() < this->getNumDimensions() ) {
      fatal("Can not deduct the region, provided reference has more " <<
            "dimensions (" << this->getNumDimensions() << ") than registered " <<
            "object (" << ref.getNumDimensions() << ").");
   } else if ( ref.getNumDimensions() > this->getNumDimensions() ) {
      unsigned int matching = 0;
      std::size_t elemSize[ ref.getNumDimensions() ];
      nanos_region_dimension_internal_t const *refDims = ref.getDimensions();
      nanos_region_dimension_internal_t const *thisDims = this->getDimensions();
      unsigned int dimIdx;
      std::size_t current_elem_size = 1;
      std::size_t current_offset = 0;
      uint64_t off = (uint64_t)this->getBaseAddress() - (uint64_t)ref.getBaseAddress();
      for ( dimIdx = 0; dimIdx < this->getNumDimensions(); dimIdx += 1) {
         if ( refDims[ dimIdx ].size == thisDims[ dimIdx ].size ) {
            matching += 1;
            newDims[ dimIdx ].size = thisDims[ dimIdx ].size;
            newDims[ dimIdx ].accessed_length = thisDims[ dimIdx ].accessed_length;
            newDims[ dimIdx ].lower_bound = thisDims[ dimIdx ].lower_bound;
            current_offset += thisDims[ dimIdx ].lower_bound * current_elem_size;
            current_elem_size *= refDims[ dimIdx ].size;
            elemSize[ dimIdx ] = current_elem_size;
            if ( off % elemSize[dimIdx] != 0 ) { 
               //for now, address must be aligned, if not, lower_bound must
               //be used to adjust the offset
               fatal("Address does not correspond with the provided lower_bound ("
                     << thisDims[dimIdx].lower_bound << ") for dimension #"
                     << dimIdx << ")." );
            }
         } else if ( refDims[ dimIdx ].size > thisDims[ dimIdx ].size ) {
            fatal("can't deduct dimensions of registered object. Specified dimension "
                  << dimIdx << " size (" << thisDims[dimIdx].size <<
                  ") does not match with registered object dimension size ("<<
                  refDims[dimIdx].size <<").");
         } else { /* refDims[ dimIdx ].size < thisDims[ dimIdx ].size */
            fatal("can't deduct dimensions of registered object. Specified dimension "
                  << dimIdx << " size (" << thisDims[dimIdx].size <<
                  ") does not match with registered object dimension size ("<<
                  refDims[dimIdx].size <<").");
         }
      }
      for ( ; dimIdx < ref.getNumDimensions(); dimIdx += 1 ) {
         current_elem_size *= refDims[ dimIdx ].size;
         elemSize[ dimIdx ] = current_elem_size;
      }

      for ( unsigned int idx = this->getNumDimensions(); idx < ref.getNumDimensions(); idx += 1 ) {
         //uint64_t toff = ((off - current_offset) % elemSize[idx]) % elemSize[ idx-1 ];
         //std::cerr << " offset: " << current_offset << " off: " << off << " elemSize(" << idx << "): " << elemSize[idx] << " elemSize(" << idx-1 << ") " << elemSize[idx-1]<< std::endl;
         uint64_t lower_bound = (off % elemSize[idx]) / elemSize[ idx-1 ];
         newDims[ idx ].size = refDims[ idx ].size;
         newDims[ idx ].lower_bound = lower_bound;
         newDims[ idx ].accessed_length = 1;
         //std::cerr << idx << " this dim lower bound: " << lower_bound << " size: " << refDims[ idx ].size << " off: "<< current_offset << std::endl;
      }
   } else {
      if ( this->getNumDimensions() == 1 ) {
         uint64_t off = (uint64_t)this->getBaseAddress() - (uint64_t)ref.getBaseAddress();
         nanos_region_dimension_internal_t const *refDims = ref.getDimensions();
         nanos_region_dimension_internal_t const *thisDims = this->getDimensions();
         newDims[0].size = refDims[0].size;
         newDims[0].lower_bound = off;
         newDims[0].accessed_length = thisDims[0].accessed_length;
         //std::cerr << "set newDims " << &(newDims[0]) << " used refDims " << refDims << " and this dims " << thisDims << std::endl;
      } else {
         message("Warning: deductCd not properly implemented when there are the same number of dimensions and more than 1 dimension.");
         //std::cerr << "Warning: deductCd not properly implemented when there are the same number of dimensions and more than 1 dimension." << std::endl;
         ::memcpy(newDims, this->getDimensions(), sizeof(nanos_region_dimension_internal_t) * this->getNumDimensions());
      }
   }
}

std::ostream& nanos::operator<< (std::ostream &o, CopyData const &cd) {
   o << "CopyData" << std::endl;
   o << "\tAddtess: " << cd.address << std::endl;
   o << "\tHostBaseAddtess (" << (void*) &cd.host_base_address << " ): " << cd.host_base_address << std::endl;
   o << "\tOffset: " << cd.offset << std::endl;
   o << "\tDimensions: " << cd.dimension_count << std::endl;
   for ( int i = 0; i < cd.dimension_count; i++) {
      o << "\t\t[" << i << "] size: " << cd.dimensions[i].size << ", accessed length: " << cd.dimensions[i].accessed_length << ", lower bound: " << cd.dimensions[i].lower_bound << std::endl;
   } 
   o << "End of CopyData" << std::endl;
   return o;
}

bool CopyData::equalGeometry( CopyData const &cd ) const {
   bool result = true;
   if ( cd.dimension_count != this->dimension_count || cd.address != this->address ) {
      result = false;
   } else {
      for ( int idx = 0; idx < this->dimension_count && result; idx += 1 ) {
         result = cd.dimensions[idx].size == this->dimensions[idx].size;
      }
   }
   return result;
}

