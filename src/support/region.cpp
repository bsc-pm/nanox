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


#include <ostream>

#include "region_decl.hpp"


std::ostream& nanos::operator<< (std::ostream &o, Region const & region) {
   o << (int)region.m_firstBit << ":\\ ";
   unsigned int currentBit = region.m_firstBit;
   for (int bitIndex = 0; bitIndex < region.getLength(); bitIndex++) {
      switch (region[bitIndex]) {
      case Region::BIT_0:
         o << "0";
         break;
      case Region::BIT_1:
         o << "1";
         break;
      case Region::X:
         o << "X";
         break;
      }
      currentBit++;
      if (currentBit % 4 == 0 && currentBit != sizeof(size_t)*8) {
         o << " ";
      }
   }

   return o;
}

