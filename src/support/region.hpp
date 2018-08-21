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


#ifndef _NANOS_REGION
#define _NANOS_REGION


#include "bitcounter.hpp"
#include "region_decl.hpp"


namespace nanos {


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

inline BaseDependency* Region::clone() const
{
   return NEW Region( *this );
}

inline void * Region::getAddress () const
{
   return (void*) m_value;
}

inline bool Region::overlap ( const BaseDependency &obj ) const
{
    const Region& region( static_cast<const Region&>( obj ) );
    return matches(region);
}

} // namespace nanos

#endif // _NANOS_REGION
