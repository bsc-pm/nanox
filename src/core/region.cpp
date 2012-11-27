/*
	Cell/SMP Superscalar (CellSs/SMPSs): Easy programming the Cell BE/Shared Memory Processors
	Copyright (C) 2008 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
	
	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.
	
	This library is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
	Lesser General Public License for more details.
	
	You should have received a copy of the GNU Lesser General Public
	License along with this library; if not, write to the Free Software
	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
	
	The GNU Lesser General Public License is contained in the file COPYING.
*/


#include <ostream>

#include "region_decl.hpp"


std::ostream& nanos::operator<< (std::ostream &o, Region const & region) {
   o << (int)region.m_firstBit << ":\\ mask " << (void*)region.m_mask << " \\ value " << (void *)region.m_value << " \\ ";
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

void nanos::Region::printSimple( std::ostream& o ) const {
   unsigned int currentBit = m_firstBit;

   //for ( unsigned int i = 0; i < currentBit; i++){ 
   //  if (i % 4 == 0 ) o << " ";
   //  o << "0";
   //}
   for (int bitIndex = 0; bitIndex < getLength(); bitIndex++) {
      switch ((*this)[bitIndex]) {
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
   o << " " << (void *) getFirstValue() << " " << (void *) ( getFirstValue() + getBreadth() );
}
