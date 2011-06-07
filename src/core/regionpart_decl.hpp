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


#ifndef _NANOS_REGION_PART_DECL
#define _NANOS_REGION_PART_DECL


#include "containeradapter.hpp"
#include "region_decl.hpp"


namespace nanos {
   
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
      int const &getPartitionLevel() const
         { return m_partLevel; }
      
      //! \brief Return the result of skiping high order bits
      //! \param bit number of high order bits to skip
      //! \returns result of skipping the bits
      RegionPart operator+(int bits) const
         { return RegionPart(*(Region *)this + bits, m_partLevel); }
      
      //! \brief Generate a region that extends this region to cover the extra elements of another without allowing aliasing
      //! \param other the other region
      //! \param[out] result the extended region
      //! \returns true if the combination is possible without aliasing
      bool combine(RegionPart const &other, /* Outputs: */ RegionPart &result) const
         {
            bool successful = Region::combine(other, /* Output */ result);
            if (successful && m_partLevel > 0) {
               result.m_partLevel = m_partLevel-1;
            } else {
               result.m_partLevel = 0;
            }
            return successful;
         }
      
      //! \brief Fill a list with the disjoint subregions determined by its intersection with another region
      //! \tparam CONTAINER_T type of the output list
      //! \param other the other region
      //! \param[out] output the output list
      //! \param maxPartitioningLevels bound on partitioning levels. By default infinite
      //! \param currentPartitionDepth recursivity control. Must be 0 on outermost call
      //! \param outputMatching determines if subregions that intersect must be added to the output. By default true
      //! \param outputNonMatching determines if subregions out of the intersection must be added to the output. By default true
      template <typename CONTAINER_T>
      void partition(Region const &other, /* Output: */ CONTAINER_T &output, int maxPartitioningLevels = -1, int currentPartitionDepth = 0, bool outputMatching=true, bool outputNonMatching=true) const
         {
            if (maxPartitioningLevels != -1 && currentPartitionDepth + m_partLevel == maxPartitioningLevels) {
               RegionPart copy(*this);
               copy.m_partLevel += currentPartitionDepth;
               if (outputMatching && outputNonMatching) {
                  ContainerAdapter<CONTAINER_T>::insert(output, copy);
               } else if (outputMatching) {
                  // !outputNonMatching
                  if (matches(other)) {
                     ContainerAdapter<CONTAINER_T>::insert(output, copy);
                  }
               } else if (outputNonMatching) {
                  // !outputMatching
                  if (!matches(other)) {
                     ContainerAdapter<CONTAINER_T>::insert(output, copy);
                  } else {
                     bool thisSubsumes = false;
                     bool otherSubsumes = false;
                     bool match = false;
                     compare(other, /* Outputs: */ match, thisSubsumes, otherSubsumes);
                     if (match && thisSubsumes) {
                        // Part of this does not match other
                        ContainerAdapter<CONTAINER_T>::insert(output, copy);
                     }
                  }
               }
               return;
            }
            
            bitfield_t const ONE = 1;
            bool subsumed = false;
            
            int shortestLengthIndex = (m_nextBit > other.m_nextBit ? m_nextBit : other.m_nextBit);
            signed char firstMismatch = (sizeof(bitfield_t) * 8) - 1;
            bitfield_t cleanerMask = 0;
            cleanerMask--;
            while (firstMismatch > shortestLengthIndex) {
               if (m_mask & (ONE << firstMismatch)) {
                  if (other.m_mask & (ONE << firstMismatch)) {
                     if ( (m_value & (ONE << firstMismatch)) != (other.m_value & (ONE << firstMismatch)) ) {
                        // The current bit is different. So there is nothing to partition
                        subsumed = false;
                        break;
                     }
                  } else {
                     // other subsumes (nothing to partition)
                  }
               } else {
                  // This has an X
                  if (other.m_mask & (ONE << firstMismatch)) {
                     // This subsumes
                     subsumed = true;
                     break;
                  }
               }
               
               firstMismatch--;
               cleanerMask = cleanerMask >> 1;
            }
            
            if (subsumed) {
               int skipBits = ((sizeof(bitfield_t) * 8) - 1) + 1 - firstMismatch;
               
               CONTAINER_T subparts;
               (*this + skipBits).partition(other + skipBits, subparts, maxPartitioningLevels, currentPartitionDepth+1);
               
               bit_value_t subsumedBit = other[skipBits - 1];
               bit_value_t complementaryBit = (subsumedBit == BIT_0 ? BIT_1 : BIT_0);
               
               if (outputNonMatching) {
                  // Output the non matching part. Since it does not match, it does not require partitioning.
                  Region nonMatchingPart;
                  nonMatchingPart.m_firstBit = m_firstBit;
                  nonMatchingPart.m_nextBit = firstMismatch;
                  nonMatchingPart.m_mask = m_mask & ~cleanerMask;
                  nonMatchingPart.m_value = m_value & ~cleanerMask;
                  nonMatchingPart.addBit(complementaryBit);
                  nonMatchingPart += *this + skipBits;
                  ContainerAdapter<CONTAINER_T>::insert(output,  RegionPart(nonMatchingPart, currentPartitionDepth + m_partLevel + 1) );
               }
               
               for (typename CONTAINER_T::iterator it = subparts.begin(); it != subparts.end(); it++) {
                  RegionPart subpart = *it;
                  RegionPart matchingSubpart;
                  matchingSubpart.m_firstBit = m_firstBit;
                  matchingSubpart.m_nextBit = firstMismatch;
                  matchingSubpart.m_mask = m_mask & ~cleanerMask;
                  matchingSubpart.m_value = m_value & ~cleanerMask;
                  matchingSubpart.addBit(subsumedBit);
                  matchingSubpart += subpart;
                  matchingSubpart.m_partLevel = subpart.m_partLevel;
                  
                  if (outputMatching && outputNonMatching) {
                     ContainerAdapter<CONTAINER_T>::insert(output, matchingSubpart);
                  } else if (outputMatching) {
                     // !outputNonMatching
                     if (matchingSubpart.matches(other)) {
                        ContainerAdapter<CONTAINER_T>::insert(output, matchingSubpart);
                     }
                  } else if (outputNonMatching) {
                     // !outputMatching
                     if (!matchingSubpart.matches(other)) {
                        ContainerAdapter<CONTAINER_T>::insert(output, matchingSubpart);
                     } else {
                        bool matchingSubpartSubsumes = false;
                        bool otherSubsumes = false;
                        bool match = false;
                        matchingSubpart.compare(other, match, matchingSubpartSubsumes, otherSubsumes);
                        if (match && matchingSubpartSubsumes) {
                           // Part of matchingSubpart does not match other
                           ContainerAdapter<CONTAINER_T>::insert(output, matchingSubpart);
                        }
                     }
                  }
               }
            } else {
               // Total match or mismatch
               bool matches_ = matches(other);
               if ( (outputMatching && matches_) || (outputNonMatching && !matches_) ) {
                  ContainerAdapter<CONTAINER_T>::insert(output,  RegionPart(*this, m_partLevel + currentPartitionDepth) );
               }
            }
         }
      
   };
   
} // namespace nanos


#endif // _NANOS_REGION_PART_DECL
