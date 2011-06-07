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


#ifndef _NANOS_REGION_COLLECTION_DECL
#define _NANOS_REGION_COLLECTION_DECL

#include <list>

#include "region.hpp"
#include "regionpart_decl.hpp"


namespace nanos {
   
   //! \brief Collection of RegionPart possibly to represent a shape that cannot be represented by a single Region
   //! \ptype container_t type of the container that stores each RegionPart
   template < class container_t = std::list<RegionPart> >
   class RegionCollection {
   protected:
      //! Contents of the collection
      container_t m_parts;
      
   public:
      typedef typename container_t::const_iterator const_iterator;
      typedef typename container_t::iterator iterator;
      
      //! \brief Default constructor
      RegionCollection(): m_parts()
         {}
      
      //! \brief One element constructor
      //! \param part the region part
      RegionCollection(RegionPart const &part): m_parts()
         { ContainerAdapter<container_t>::insert(m_parts, part); }
      
      //! \brief Copy constructor
      //! \param other the other RegionCollection
      RegionCollection(RegionCollection const &other): m_parts(other.m_parts)
         {}
      
      //! \brief Destructor
      ~RegionCollection()
         { m_parts.clear(); }
      
      //! \brief Add a RegionPart
      //! \param part the RegionPart
      void addPart(RegionPart const &part)
         { ContainerAdapter<container_t>::insert(m_parts, part); }
      
      //! \brief Add a Region
      //! \param region the Region
      void addPart(Region const &region)
         { ContainerAdapter<container_t>::insert(m_parts, RegionPart(region, 0)); }
      
      //! \brief Number of elements
      //! \return the number of elements
      size_t getSize() const
         { return m_parts.getSize(); }
      
      //! \brief Check if it is empty
      //! \returns true if it is empty
      bool empty() const
         { return m_parts.empty(); }
      
      //! \brief Calculate result of removing the intersection with a Region
      //! \param substracted the Region whose intersections must be removed
      //! \param maxPartitioningLevels the maximum fragmentation level allowed. By default infinite. If specified, some intersecting subparts may remain
      //! \returns the result of the removal
      RegionCollection minus(Region const &substracted, int maxPartitioningLevels = -1) const
         {
            RegionCollection result;
            for (typename container_t::const_iterator it = m_parts.begin(); it != m_parts.end(); it++) {
               RegionPart const &part = *it;
               part.partition(substracted, result.m_parts, maxPartitioningLevels, 0, false, true);
            }
            // result.defragment();
            return result;
         }
      
      //! \brief Calculate the result of removing the intersection with a RegionCollection
      //! \param substracted the RegionCollection whose intersections must be removed
      //! \param maxPartitioningLevels the maximum fragmentation level allowed. By default infinite. If specified, some intersecting subparts may remain
      //! \returns the result of the removal
      RegionCollection minus(RegionCollection const &substracted, int maxPartitioningLevels = -1) const
         {
            RegionCollection result(*this);
            for (typename container_t::const_iterator it = substracted.m_parts.begin(); it != substracted.m_parts.end(); it++) {
               RegionPart const &part = *it;
               result.minus(part, maxPartitioningLevels);
            }
            return result;
         }
      
      //! \brief Remove the intersection with a Region
      //! \param substracted the Region whose intersections must be removed
      //! \param maxPartitioningLevels the maximum fragmentation level allowed. By default infinite. If specified, some intersecting subparts may remain
      //! \returns itself
      RegionCollection &substract(Region const &substracted, int maxPartitioningLevels = -1)
         {
            (*this) = minus(substracted, maxPartitioningLevels);
            return *this;
         }
      
      //! \brief Remove the intersection with a RegionCollection
      //! \param substracted the RegionCollection whose intersections must be removed
      //! \param maxPartitioningLevels the maximum fragmentation level allowed. By default infinite. If specified, some intersecting subparts may remain
      //! \returns itself
      RegionCollection &substract(RegionCollection const &substracted, int maxPartitioningLevels = -1)
         {
            (*this) = minus(substracted, maxPartitioningLevels);
            return *this;
         }
      
      //! \brief Extends each RegionPart to cover the extra elements of each RegionPart in the other RegionPartCollection without allowing aliasing
      //! \param other the RegionCollection RegionParts will be used to extend its RegionParts
      //! \returns itself
      RegionCollection &operator|=(RegionCollection const &other)
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
      
      //! \brief Const reference to the contents
      //! \returns a const reference to the contents
      container_t const &getRegionCollection() const
         { return m_parts; }
      
      //! \brief Begining of constant iteration
      //! \returns a const_iterator pointing to the first element
      const_iterator begin() const
         { return m_parts.begin(); }
      
      //! \brief Begining of iteration
      //! \returns an iterator pointing to the first element
      iterator begin()
         { return m_parts.begin(); }
      
      //! \brief End of constant iteration
      //! \returns a const_iterator pointing to the end of the container
      const_iterator end() const
         { return m_parts.end(); }
      
      //! \brief End of iteration
      //! \returns an iterator pointing to the end of the container
      iterator end()
         { return m_parts.end(); }
      
      //! \brief Modificable reference to the contents
      //! \returns a modificable reference to the contents
      container_t &getRegionCollectionReference()
         { return m_parts; }
      
      //! \brief Join parts that can be represented with one RegionPart without adding aliasing
      void defragment()
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

   };

} // namespace nanos


#endif // _NANOS_REGION_COLLECTION_DECL
