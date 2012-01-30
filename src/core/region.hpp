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


#ifndef _NANOS_REGION
#define _NANOS_REGION


#include "bitcounter.hpp"
#include "region_decl.hpp"


using namespace nanos;


inline bool Region::combine(Region const &other, /* Outputs: */ Region &result) const
{
   if (m_firstBit != other.m_firstBit || m_nextBit != other.m_nextBit) {
      return false;
   }
   
   if ((m_mask != (m_mask | other.m_mask)) && (other.m_mask != (m_mask | other.m_mask))) {
      // The masks partially subsume each other
      return false;
   }
   
   if ((m_mask != other.m_mask) && ((m_value & m_mask & other.m_mask) != (other.m_value & m_mask & other.m_mask))) {
      // The masks are different but the values are different too
      return false;
   }
   
   // We'll operate directly over the result
   result = (*this);
   
   result.m_mask = result.m_mask & other.m_mask; // Combine the masks (NOTE: mask is corrected at the end)
   bitfield_t valueDifferences = (result.m_value ^ other.m_value) & result.m_mask;
   
   // Check whether there is more than one different bit in the value
   if (BitCounter<bitfield_t>::hasMoreThanOneOne(valueDifferences)) {
      // More than one difference -> if we combined them we would be including other strings
      return false;
   }
   
   // Mark the different bit (if present) as X
   result.m_mask = result.m_mask & ~valueDifferences;
   result.m_value = result.m_value & result.m_mask; // This also corrects the previous effect of combining the masks
   
   return true;
}


#endif // _NANOS_REGION
