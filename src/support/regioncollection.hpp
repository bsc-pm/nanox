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


#ifndef _NANOS_REGION_COLLECTION
#define _NANOS_REGION_COLLECTION

#include <list>

#include "regioncollection_decl.hpp"
#include "containeradapter.hpp"
#include "regionpart.hpp"

namespace nanos {

template<class container_t>
inline void RegionCollection<container_t>::addPart(RegionPart const &part)
{
   ContainerAdapter<container_t>::insert(m_parts, part);
}

template<class container_t>
inline void RegionCollection<container_t>::addPart(Region const &region)
{
   ContainerAdapter<container_t>::insert(m_parts, RegionPart(region, 0));
}

template<class container_t>
inline size_t RegionCollection<container_t>::getSize() const
{
   return m_parts.getSize();
}

template<class container_t>
inline bool RegionCollection<container_t>::empty() const
{
   return m_parts.empty();
}


template<class container_t>
inline RegionCollection<container_t> RegionCollection<container_t>::minus(Region const &substracted, int maxPartitioningLevels) const
{
   RegionCollection result;
   for (typename container_t::const_iterator it = m_parts.begin(); it != m_parts.end(); it++) {
      RegionPart const &part = *it;
      part.partition(substracted, result.m_parts, maxPartitioningLevels, 0, false, true);
   }
   // result.defragment();
   return result;
}

template<class container_t>
inline RegionCollection<container_t> RegionCollection<container_t>::minus(RegionCollection const &substracted, int maxPartitioningLevels) const
{
   RegionCollection result(*this);
   for (typename container_t::const_iterator it = substracted.m_parts.begin(); it != substracted.m_parts.end(); it++) {
      RegionPart const &part = *it;
      result = result.minus(part, maxPartitioningLevels);
   }
   return result;
}

template<class container_t>
inline RegionCollection<container_t>& RegionCollection<container_t>::substract(Region const &substracted, int maxPartitioningLevels)
{
   (*this) = minus(substracted, maxPartitioningLevels);
   return *this;
}

template<class container_t>
inline RegionCollection<container_t>& RegionCollection<container_t>::substract(RegionCollection const &substracted, int maxPartitioningLevels)
{
   (*this) = minus(substracted, maxPartitioningLevels);
   return *this;
}

template<class container_t>
inline RegionCollection<container_t>& RegionCollection<container_t>::operator|=(RegionCollection const &other)
{
   RegionCollection ours(*this);
   m_parts.clear();
   
   for (const_iterator it = ours.begin(); it != ours.end(); it++) {
      RegionPart const &ourPart = *it;
      for (const_iterator it2 = other.begin(); it2 != other.end(); it2++) {
         RegionPart newPart = *it2;
         newPart |= ourPart;
         addPart(newPart);
      }
   }
   defragment();
   
   return *this;
}

template<class container_t>
inline container_t const & RegionCollection<container_t>::getRegionCollection() const
{
   return m_parts;
}

template<class container_t>
inline typename RegionCollection<container_t>::const_iterator RegionCollection<container_t>::begin() const
{
   return m_parts.begin();
}

template<class container_t>
inline typename RegionCollection<container_t>::iterator RegionCollection<container_t>::begin()
{
   return m_parts.begin();
}

template<class container_t>
inline typename RegionCollection<container_t>::const_iterator RegionCollection<container_t>::end() const
{
   return m_parts.end();
}

template<class container_t>
inline typename RegionCollection<container_t>::iterator RegionCollection<container_t>::end()
{
   return m_parts.end();
}

template<class container_t>
inline container_t & RegionCollection<container_t>::getRegionCollectionReference()
{
   return m_parts;
}

template<class container_t>
inline container_t const & RegionCollection<container_t>::getRegionCollectionReference() const
{
   return m_parts;
}

template<class container_t>
inline void RegionCollection<container_t>::defragment()
{
   container_t regions(m_parts);
   container_t regions2;
   
   bool effective;
   do {
      effective = false;
      typename container_t::iterator it = regions.begin();
      while (it != regions.end()) {
         if (!it->isEmpty()) {
            bool found = false;
            typename container_t::iterator it2 = it;
            it2++;
            while (it2 != regions.end()) {
               if (!it2->isEmpty()) {
                  RegionPart combinedRegion;
                  if (it->combine(*it2, /* out */ combinedRegion)) {
                     found = true;
                     effective = true;
                     it->clear();
                     it2->clear();
                     regions2.push_back(combinedRegion);
                     break;
                  }
               }
               it2++;
            }
            
            if (!found) {
               regions2.push_back(*it);
            }
         }
         it++;
      }
      
      if (effective) {
         regions.clear();
         regions.splice(regions.begin(), regions2);
         // regions2.clear();
      }
   } while (effective);
   
   m_parts.clear();
   m_parts.splice(m_parts.begin(), regions2);
   
   regions.clear();
   // regions2.clear();
}
   
template <class container_t>
std::ostream& operator<< (std::ostream& o, RegionCollection<container_t> const &regionCollection)
{
   container_t const &contents = regionCollection.getRegionCollectionReference();
   for (typename container_t::const_iterator it = contents.begin(); it != contents.end(); it++) {
      RegionPart const &part = *it;
      
      o << "\t" << part << std::endl;
   }
   
   return o;
}

} // namespace nanos

#endif // _NANOS_REGION_COLLECTION
