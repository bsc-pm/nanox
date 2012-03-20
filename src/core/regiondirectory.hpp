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

#ifndef _NANOS_NEWDIRECTORY_H
#define _NANOS_NEWDIRECTORY_H

#include "regiondirectory_decl.hpp"
#include "regionbuilder.hpp"

template <class RegionDesc>
Region NewRegionDirectory::build_region( RegionDesc const &dataAccess ) {
   // Find out the displacement due to the lower bounds and correct it in the address
   size_t base = 1UL;
   size_t displacement = 0L;
   for (short dimension = 0; dimension < dataAccess.dimension_count; dimension++) {
      nanos_region_dimension_internal_t const &dimensionData = dataAccess.dimensions[dimension];
      displacement = displacement + dimensionData.lower_bound * base;
      base = base * dimensionData.size;
   }
   size_t address = (size_t)dataAccess.address + displacement;

   // Build the Region

   // First dimension is base 1
   size_t additionalContribution = 0UL; // Contribution of the previous dimensions (necessary due to alignment issues)
   //std::cerr << "build region 0 len is " << dataAccess.dimensions[0].accessed_length << std::endl;
   Region region = RegionBuilder::build(address, 1UL, dataAccess.dimensions[0].accessed_length, additionalContribution);

   // Add the bits corresponding to the rest of the dimensions (base the previous one)
   base = 1 * dataAccess.dimensions[0].size;
   for (short dimension = 1; dimension < dataAccess.dimension_count; dimension++) {
      nanos_region_dimension_internal_t const &dimensionData = dataAccess.dimensions[dimension];

      //std::cerr << "build region " << dimension << " len is " << dimensionData.accessed_length << std::endl;
      region |= RegionBuilder::build(address, base, dimensionData.accessed_length, additionalContribution);
      base = base * dimensionData.size;
   }
   //std::cerr << "end build region n" << std::endl;

   return region;
}

template <class RegionDesc>
Region NewRegionDirectory::build_region_with_given_base_address( RegionDesc const &dataAccess, uint64_t newBaseAddress ) {
   // Find out the displacement due to the lower bounds and correct it in the address
   size_t base = 1UL;
   size_t displacement = 0L;
   for (short dimension = 0; dimension < dataAccess.dimension_count; dimension++) {
      nanos_region_dimension_internal_t const &dimensionData = dataAccess.dimensions[dimension];
      displacement = displacement + dimensionData.lower_bound * base;
      base = base * dimensionData.size;
   }
   size_t address = (size_t)newBaseAddress + displacement;

   // Build the Region

   // First dimension is base 1
   size_t additionalContribution = 0UL; // Contribution of the previous dimensions (necessary due to alignment issues)
   //std::cerr << "build region 0 len is " << dataAccess.dimensions[0].accessed_length << std::endl;
   Region region = RegionBuilder::build(address, 1UL, dataAccess.dimensions[0].accessed_length, additionalContribution);

   // Add the bits corresponding to the rest of the dimensions (base the previous one)
   base = 1 * dataAccess.dimensions[0].size;
   for (short dimension = 1; dimension < dataAccess.dimension_count; dimension++) {
      nanos_region_dimension_internal_t const &dimensionData = dataAccess.dimensions[dimension];

      //std::cerr << "build region " << dimension << " len is " << dimensionData.accessed_length << std::endl;
      region |= RegionBuilder::build(address, base, dimensionData.accessed_length, additionalContribution);
      base = base * dimensionData.size;
   }
   //std::cerr << "end build region n" << std::endl;

   return region;
}
#endif
