/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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


#ifndef _NANOS_REGION_COLLECTION_DECL
#define _NANOS_REGION_COLLECTION_DECL

#include <list>

#include "containeradapter_decl.hpp"
#include "regioncollection_fwd.hpp"
#include "regionpart_decl.hpp"


namespace nanos
{
   
   //! \brief Region stream formatter
   //! \param o the output stream
   //! \tparam container_t the type of the collection container
   //! \param regionCollection the region collection
   //! \returns the output stream
   template <class container_t>
   std::ostream& operator<< (std::ostream& o, nanos::RegionCollection<container_t> const &regionCollection);
   
   
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
      void addPart(RegionPart const &part);
      
      //! \brief Add a Region
      //! \param region the Region
      void addPart(Region const &region);
      
      //! \brief Number of elements
      //! \return the number of elements
      size_t getSize() const;
      
      //! \brief Check if it is empty
      //! \returns true if it is empty
      bool empty() const;
      
      //! \brief Calculate result of removing the intersection with a Region
      //! \param substracted the Region whose intersections must be removed
      //! \param maxPartitioningLevels the maximum fragmentation level allowed. By default infinite. If specified, some intersecting subparts may remain
      //! \returns the result of the removal
      RegionCollection minus(Region const &substracted, int maxPartitioningLevels = -1) const;
      
      //! \brief Calculate the result of removing the intersection with a RegionCollection
      //! \param substracted the RegionCollection whose intersections must be removed
      //! \param maxPartitioningLevels the maximum fragmentation level allowed. By default infinite. If specified, some intersecting subparts may remain
      //! \returns the result of the removal
      RegionCollection minus(RegionCollection const &substracted, int maxPartitioningLevels = -1) const;
      
      //! \brief Remove the intersection with a Region
      //! \param substracted the Region whose intersections must be removed
      //! \param maxPartitioningLevels the maximum fragmentation level allowed. By default infinite. If specified, some intersecting subparts may remain
      //! \returns itself
      RegionCollection &substract(Region const &substracted, int maxPartitioningLevels = -1);
      
      //! \brief Remove the intersection with a RegionCollection
      //! \param substracted the RegionCollection whose intersections must be removed
      //! \param maxPartitioningLevels the maximum fragmentation level allowed. By default infinite. If specified, some intersecting subparts may remain
      //! \returns itself
      RegionCollection &substract(RegionCollection const &substracted, int maxPartitioningLevels = -1);
      
      //! \brief Extends each RegionPart to cover the extra elements of each RegionPart in the other RegionPartCollection without allowing aliasing
      //! \param other the RegionCollection RegionParts will be used to extend its RegionParts
      //! \returns itself
      RegionCollection &operator|=(RegionCollection const &other);
      
      //! \brief Const reference to the contents
      //! \returns a const reference to the contents
      container_t const &getRegionCollection() const;
      
      //! \brief Begining of constant iteration
      //! \returns a const_iterator pointing to the first element
      const_iterator begin() const;
      
      //! \brief Begining of iteration
      //! \returns an iterator pointing to the first element
      iterator begin();
      
      //! \brief End of constant iteration
      //! \returns a const_iterator pointing to the end of the container
      const_iterator end() const;
      
      //! \brief End of iteration
      //! \returns an iterator pointing to the end of the container
      iterator end();
      
      //! \brief Modificable reference to the contents
      //! \returns a modificable reference to the contents
      container_t &getRegionCollectionReference();
      
      //! \brief Unmodificable reference to the contents
      //! \returns a unmodificable reference to the contents
      container_t const &getRegionCollectionReference() const;
      
      //! \brief Join parts that can be represented with one RegionPart without adding aliasing
      void defragment();
      
   };
   
   template <class container_t>
   std::ostream& operator<< (std::ostream& o, RegionCollection<container_t> const &regionCollection);
   
} // namespace nanos


#endif // _NANOS_REGION_COLLECTION_DECL
