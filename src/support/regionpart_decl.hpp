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


#ifndef _NANOS_REGION_PART_DECL
#define _NANOS_REGION_PART_DECL

#include "region_decl.hpp"


namespace nanos
{
   
   /*! \class RegionPart
   *  \brief A low-level representation of a region with partitioning information
   */
   class RegionPart: public Region {
   private:
      //! Paritioning level
      int m_partLevel;
      
   public:
      //! \brief Default constructor
      RegionPart(): Region(), m_partLevel(0)
         {}
      
      //! \brief Constructor by parts
      RegionPart(Region const &region, int partLevel): Region(region), m_partLevel(partLevel)
         {}
      
      //! \brief Copy constructor
      RegionPart(RegionPart const &other): Region(other), m_partLevel(other.m_partLevel)
         {}
      
      //! \brief Partitioning level
      //! \returns the region partitioning level
      int const &getPartitionLevel() const;
      
      //! \brief Return the result of skiping high order bits
      //! \param bit number of high order bits to skip
      //! \returns result of skipping the bits
      RegionPart operator+(int bits) const;
      
      //! \brief Generate a region that extends this region to cover the extra elements of another without allowing aliasing
      //! \param other the other region
      //! \param[out] result the extended region
      //! \returns true if the combination is possible without aliasing
      bool combine(RegionPart const &other, /* Outputs: */ RegionPart &result) const;
      
      //! \brief Fill a list with the disjoint subregions determined by its intersection with another region
      //! \tparam CONTAINER_T type of the output list
      //! \param other the other region
      //! \param[out] output the output list
      //! \param maxPartitioningLevels bound on partitioning levels. By default infinite
      //! \param currentPartitionDepth recursivity control. Must be 0 on outermost call
      //! \param outputMatching determines if subregions that intersect must be added to the output. By default true
      //! \param outputNonMatching determines if subregions out of the intersection must be added to the output. By default true
      template <typename CONTAINER_T>
      void partition(Region const &other, /* Output: */ CONTAINER_T &output, int maxPartitioningLevels = -1, int currentPartitionDepth = 0, bool outputMatching=true, bool outputNonMatching=true) const;      
   };
   
} // namespace nanos


#endif // _NANOS_REGION_PART_DECL
