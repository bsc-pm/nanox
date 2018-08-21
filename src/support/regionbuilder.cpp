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


//#include <cstddef>
#include <stddef.h>

#include "region.hpp"
#include "regionbuilder_decl.hpp"


namespace nanos {


Region RegionBuilder::build(size_t address, size_t base, size_t length, size_t /* INOUT */ &additionalContribution) {
   Region::bitfield_t const ONE = 1;
   Region::bitfield_t const ALL_ONE = ~((Region::bitfield_t)0);
   size_t const wordLength = sizeof(size_t)*8UL;
   
   //
   // Calculate the lowest bits of the mask (base mask relative to address 0).
   // That is, the part of the mask due to the stride determined by the mask.
   //
   size_t lowMask;
   
   if (base != 1) {
      // Find where the skiped bits begin
      lowMask = base - 1;
      size_t maskLowest1;
      for (maskLowest1 = 0; maskLowest1 <= wordLength-1UL; maskLowest1++) {
         if (lowMask & (ONE << maskLowest1)) {
            break;
         }
      }
      
      
      // Mark the following bits
      lowMask = lowMask | ((ONE << maskLowest1) - 1);
      lowMask = ~lowMask;
   } else {
      lowMask = ALL_ONE;
   }
   
   size_t lastAddress = address + (length-1)*base + additionalContribution;
   
   additionalContribution = additionalContribution + (length-1UL)*base;
   
   size_t highMask=0;
   for (int bit=wordLength-1; bit >= 0; bit--) {
      size_t bitSelector = ONE << bit;
      if ((address & bitSelector) == (lastAddress & bitSelector)) {
         highMask |= bitSelector;
      } else {
         break;
      }
   }
   highMask = ~highMask;
   
   size_t mask = ~(lowMask & highMask);
   
   address = address & mask;
   
   return Region(address, mask);
}


} // namespace nanos

