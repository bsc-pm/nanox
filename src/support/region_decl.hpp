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

#ifndef _NANOS_REGION_DECL
#define _NANOS_REGION_DECL

#include <stddef.h>
#include <unistd.h>
#include <ostream>

#include "region_fwd.hpp"
#include "basedependency_decl.hpp"

namespace nanos
{
   //! \brief Region stream formatter
   //! \param o the output stream
   //! \param region the region
   //! \returns the output stream
   std::ostream& operator<< (std::ostream& o, nanos::Region const &region);


   /*! \class Region
   *  \brief A low-level representation of a region
   */
   class Region : public BaseDependency {
   protected:
      //! Type of each part of the representation
      typedef size_t bitfield_t;
      
      //! First logical bit. However, physically we start at the highest bit in every bitfield_t
      unsigned char m_firstBit;
      //! Next physical bit to be filled starting from the highest one
      signed char m_nextBit;
      //! Region mask. Indicates if which bits of m_value must be checked (1) and which ones must be ignored (0).
      bitfield_t m_mask;
      //! Region base bits. Indicates the value (0/1) of non X bits. X bits must be set to 0.
      bitfield_t m_value;
      
      friend class RegionBuilder;
      
   public:
      //! Type of a bit
      enum bit_value_t {
         BIT_0 = 0,
         BIT_1 = 1,
         X = 2
      };
      
      //! \brief Region default constructor
      Region():
         m_firstBit(0), m_nextBit((sizeof(bitfield_t)*8)-1), m_mask(0), m_value(0)
         {}
      
      //! \brief Region copy constructor
      //! \param other another Region
      Region(Region const &other):
         BaseDependency(), m_firstBit(other.m_firstBit), m_nextBit(other.m_nextBit), m_mask(other.m_mask), m_value(other.m_value)
         {}
      
      //! \brief Region constructor by parts
      //! \param value the region value part
      //! \param mask the region mask
      Region(size_t value, size_t mask):
         m_firstBit(0), m_nextBit(-1), m_mask((bitfield_t) mask), m_value((bitfield_t) value)
         {}
      
      //! \brief Check if it is initialized/valid.
      bool isEmpty() const
         { return m_firstBit == 0 && m_nextBit == (sizeof(bitfield_t)*8)-1; }
      
      //! \brief Make it empty
      void clear()
         {
            m_firstBit = 0;
            m_nextBit = (sizeof(bitfield_t)*8)-1;
            m_mask = 0;
            m_value = 0;
         }
      
      //! \brief Set the index of the first bit
      void setFirstBit(int firstBit)
         { m_firstBit = firstBit; }
      
      //! \brief Append a bit
      void addBit(bool mask, bool value)
         {
            bitfield_t maskField = ((bitfield_t)mask) << m_nextBit;
            m_mask = (m_mask & ~maskField) | maskField;
            bitfield_t valueField = ((bitfield_t)value) << m_nextBit;
            m_value = (m_value & ~valueField) | valueField;
            m_nextBit--;
         }
      //! \brief Append a bit
      void addBit(bit_value_t combinedValue)
         {
            switch (combinedValue) {
            case BIT_0:
               addBit(1, 0);
               break;
            case BIT_1:
               addBit(1, 1);
               break;
            case X:
               addBit(0, 0);
               break;
            }
         }
      
      //! \brief Skip high order bits 
      //! \param bit number of high order bits to skip
      void advance(int bits=1)
         {
            m_mask = m_mask << bits;
            m_value = m_value << bits;
            m_firstBit = m_firstBit + bits;
            m_nextBit = m_nextBit + bits;
         }
      
      //! \brief Return the result of appeding a bit
      //! \param bit the value of the bit
      //! \returns result of appending the bit
      Region operator+(bit_value_t bit) const
         {
            Region result(*this);
            result.addBit(bit);
            return result;
         }
      
      //! \brief Return the result of appeding a bit
      //! \param bit number of high order bits to skip
      //! \returns result of skipping the bits
      Region operator+(int bits) const
         {
            Region result(*this);
            result.advance(bits);
            return result;
         }
      
      //! \brief Append a region segment
      //! \param other a region segment
      void operator+=(Region const &other)
         {
            m_mask = m_mask + (other.m_mask >> getLength());
            m_value = m_value + (other.m_value >> getLength());
            m_nextBit = m_nextBit - other.getLength();
         }
      
      //! \brief Return the result of trimming a number of low order bits
      //! \param bit number of low order bits to trim
      //! \returns result of trimming the bits
      Region trim(int bits) const
         {
            bitfield_t const ONE = 1;
            bitfield_t cleanerMask = ONE << bits;
            cleanerMask--;
            bitfield_t value = m_value & ~cleanerMask;
            bitfield_t mask = m_mask & ~cleanerMask;
            Region result(value, mask);
            result.m_firstBit = m_firstBit;
            result.m_nextBit = m_nextBit + bits;
            return result;
         }
      
      //! \brief Return the result of (inclusively) trimming low order bits starting from a position
      //! \param bit first low order bit to trim
      //! \returns result of trimming the bits
      Region trimFrom(int from) const
         { return trim(getFirstBitNumber() + getLength() - from); }
      
      //! \brief Check if it intersects another region
      //! \param other the other region
      //! \returns true if it intersects the other region
      bool matches(Region const &other) const
         {
            bitfield_t const ZERO = 0;
            
            // Size check
            if (m_firstBit != other.m_firstBit || m_nextBit != other.m_nextBit) {
               return false;
            }
            
            // X check
            if ( (m_mask & other.m_mask) == ZERO ) {
               // Totally subsumed
               return true;
            }
            
            // Non X check
            return (m_value & m_mask & other.m_mask) == (other.m_value & m_mask & other.m_mask);
         }
      
      //! \brief Check if it intersects another region in their commonly diffined parts
      //! \param other the other region
      //! \returns true if they intersect in their commonly diffined parts
      bool containedMatch(Region const &other) const
         {
            bitfield_t const ONE = 1;
            bitfield_t const ZERO = 0;
            
            int shortestLengthIndex = (m_nextBit > other.m_nextBit ? m_nextBit : other.m_nextBit);
            bitfield_t cleanerMask = ( shortestLengthIndex == (sizeof(bitfield_t)*8)-1 ? 0 : ONE << (shortestLengthIndex+1) );
            cleanerMask--;
            
            bitfield_t cleanMask1 = m_mask & ~cleanerMask;
            bitfield_t cleanMask2 = other.m_mask & ~cleanerMask;
            
            // X check
            if ( (cleanMask1 & cleanMask2) == ZERO ) {
               // Totally subsumed
               return true;
            }
            
            bitfield_t cleanValue1 = m_value & ~cleanerMask;
            bitfield_t cleanValue2 = other.m_value & ~cleanerMask;
            
            // Non X check
            return (cleanValue1 & cleanMask1 & cleanMask2) == (cleanValue2 & cleanMask1 & cleanMask2);
         }
      
      //! \brief Check if it is equal to another region in their commonly diffined parts
      //! \param other the other region
      //! \returns true if they are equal in their commonly diffined parts
      bool containedExactMatch(Region const &other) const
         {
            bitfield_t const ONE = 1;
            bitfield_t const ZERO = 0;
            
            int shortestLengthIndex = (m_nextBit > other.m_nextBit ? m_nextBit : other.m_nextBit);
            bitfield_t cleanerMask = ( shortestLengthIndex == (sizeof(bitfield_t)*8)-1 ? ZERO : ONE << (shortestLengthIndex+1) );
            cleanerMask--;
            
            bitfield_t cleanMask1 = m_mask & ~cleanerMask;
            bitfield_t cleanMask2 = other.m_mask & ~cleanerMask;
            if (cleanMask1 != cleanMask2) {
               return false;
            }
            
            bitfield_t cleanValue1 = m_value & ~cleanerMask;
            bitfield_t cleanValue2 = other.m_value & ~cleanerMask;
            
            if (cleanValue1 != cleanValue2) {
               return false;
            }
            
            return true;
         }
      
      //! \brief Check if it is equal to another region
      //! \param other the other region
      //! \returns true if they are equal
      bool operator==(Region const &other) const
         {
            // Size check
            if (m_firstBit != other.m_firstBit || m_nextBit != other.m_nextBit) {
               return false;
            }
            
            // X check
            if (m_mask != other.m_mask) {
               return false;
            }
            
            // Non X check
            return (m_value == other.m_value);
         }
      
      //! \brief Check if it is different than another region
      //! \param other the other region
      //! \returns true if they are different
      bool operator!=(Region const &other) const
         { return !( (*this) == other); }
      
      //! \brief Full length complex comparison
      //! \param other the other region
      //! \param[out] matches_ true if they intersect
      //! \param[out] subsumes true if it subsumes other
      void compare(Region const &other, bool &matches_, bool &subsumes) const
         {
            bitfield_t const ZERO = 0;
            
            // Size check
            if (m_firstBit != other.m_firstBit || m_nextBit != other.m_nextBit) {
               matches_ = false;
               subsumes = false;
               return;
            }
            
            // X check
            if (m_mask != other.m_mask) {
               if ( (m_mask & other.m_mask) == ZERO ) {
                  // Totally subsumed
                  subsumes = true;
                  matches_ = true;
                  return;
               }
            }
            
            // Non X check
            if (m_value == other.m_value) {
               matches_ = true;
               subsumes = false;
            } else {
               if ( (m_value & m_mask & other.m_mask) == (other.m_value & m_mask & other.m_mask) ) {
                  matches_ = true;
                  subsumes = true;
               } else {
                  matches_ = false;
                  subsumes = false;
               }
            }
            
            return;
         }
      
      //! \brief Common defined part full comparison
      //! \param other the other region
      //! \param[out] matches_ true if their commonly defined parts intersect
      //! \param[out] thisSubsumes true if it subsumes other in their commonly defined parts
      //! \param[out] otherSubsumes true if the other subsumes it in their commonly defined parts
      void containedCompare(Region const &other, bool &matches_, bool &thisSubsumes, bool &otherSubsumes) const
         {
            bitfield_t const ZERO = 0;
            bitfield_t const ONE = 1;
            
            int shortestLengthIndex = (m_nextBit > other.m_nextBit ? m_nextBit : other.m_nextBit);
            bitfield_t cleanerMask = ( shortestLengthIndex == (sizeof(bitfield_t)*8)-1 ? 0 : ONE << (shortestLengthIndex+1) );
            cleanerMask--;
            
            bitfield_t cleanMask1 = m_mask & ~cleanerMask;
            bitfield_t cleanMask2 = other.m_mask & ~cleanerMask;
            bitfield_t cleanValue1 = m_value & ~cleanerMask;
            bitfield_t cleanValue2 = other.m_value & ~cleanerMask;
            
            // X check
            if (cleanMask1 != cleanMask2 && (cleanMask1 & cleanMask2) == ZERO ) {
               // Totally subsumed
               bitfield_t difference = cleanMask1 ^ cleanMask2;
               thisSubsumes = ~cleanMask1 & difference;
               otherSubsumes = ~cleanMask2 & difference;
               matches_ = true;
               return;
            }
            
            // Non X check
            if ( (cleanValue1 & cleanMask1 & cleanMask2) == (cleanValue2 & cleanMask1 & cleanMask2) ) {
               if (cleanMask1 == cleanMask2) {
                  matches_ = true;
                  thisSubsumes = false;
                  otherSubsumes = false;
               } else {
                  matches_ = true;
                  bitfield_t difference = cleanMask1 ^ cleanMask2;
                  thisSubsumes = ~cleanMask1 & difference;
                  otherSubsumes = ~cleanMask2 & difference;
               }
            } else {
               matches_ = false;
               thisSubsumes = false;
               otherSubsumes = false;
            }
            
            return;
         }
      
      //! \brief Full comparison
      //! \param other the other region
      //! \param[out] matches_ true if they intersect
      //! \param[out] thisSubsumes true if it subsumes
      //! \param[out] otherSubsumes true if the other subsumes
      void compare(Region const &other, /* Outputs: */ bool &matches_, bool &thisSubsumes, bool &otherSubsumes) const
         {
            bitfield_t const ZERO = 0;
            bitfield_t const ONE = 1;
            
            bitfield_t cleanerMask = ONE << (m_nextBit+1);
            cleanerMask--;
            
            bitfield_t cleanMask1 = m_mask & ~cleanerMask;
            bitfield_t cleanMask2 = other.m_mask & ~cleanerMask;
            bitfield_t cleanValue1 = m_value & ~cleanerMask;
            bitfield_t cleanValue2 = other.m_value & ~cleanerMask;
            
            // X check
            if (cleanMask1 != cleanMask2 && (cleanMask1 & cleanMask2) == ZERO ) {
               // Totally subsumed
               bitfield_t difference = cleanMask1 ^ cleanMask2;
               thisSubsumes = ~cleanMask1 & difference;
               otherSubsumes = ~cleanMask2 & difference;
               matches_ = true;
               return;
            }
            
            // Non X check
            if ( (cleanValue1 & cleanMask1 & cleanMask2) == (cleanValue2 & cleanMask1 & cleanMask2) ) {
               if (cleanMask1 == cleanMask2) {
                  matches_ = true;
                  thisSubsumes = false;
                  otherSubsumes = false;
               } else {
                  matches_ = true;
                  bitfield_t difference = cleanMask1 ^ cleanMask2;
                  thisSubsumes = ~cleanMask1 & difference;
                  otherSubsumes = ~cleanMask2 & difference;
               }
            } else {
               matches_ = false;
               thisSubsumes = false;
               otherSubsumes = false;
            }
            
            return;
         }
      
      //! \brief Check if it contains another region
      //! \param other the other region
      //! \returns true if it contains the other
      bool contains(Region const &other) const
         {
            bool matches_;
            bool thisSubsumes;
            bool otherSubsumes;
            compare(other, matches_, thisSubsumes, otherSubsumes);
            return matches_ && !otherSubsumes;
         }
      
      //! \brief Check if the first bit intersects a specific value
      //! \param bitValue the bit value
      //! \returns true if the first bit intersects the bit value
      bool firstBitMatches(bit_value_t bitValue) const
         {
            bitfield_t const ONE = 1;
            
            bitfield_t highestBitMask = ONE << (sizeof(bitfield_t) * 8 - 1);
            
            // X check
            if (bitValue == X || (~m_mask & highestBitMask)) {
               // Totally subsumed
               return true;
            }
            
            // Non X check
            if (bitValue == BIT_1) {
               return (m_value & highestBitMask);
            } else {
               return !(m_value & highestBitMask);
            }
         }
      
      //! \brief Length in bits
      //! \returns its length in bits
      int getLength() const
         { return ((sizeof(bitfield_t) * 8) - 1) - m_nextBit; }
      
      //! \brief Value of a certain bit
      //! \param index the bit index from most significant (0) to least significant
      //! \returns the value of bit index
      bit_value_t operator[](off_t index) const
         {
            bitfield_t const ONE = 1;
            
            if (m_mask & (ONE << ((sizeof(bitfield_t) * 8) - 1 - index) ) ) {
               if (m_value & (ONE << ((sizeof(bitfield_t) * 8) - 1 - index) ) ) {
                  return BIT_1;
               } else {
                  return BIT_0;
               }
            } else {
               return X;
            }
         }
      
      //! \brief Change a certain bit to 0
      //! \param index the bit index from most significant (0) to least significant
      void changeBitTo0(off_t index)
         {
            bitfield_t const ONE = 1;
            bitfield_t cleanerMask = ONE << (sizeof(bitfield_t)*8 - 1 - index);
            m_value = m_value & ~cleanerMask;
            m_mask = m_mask | cleanerMask;
         }
      
      //! \brief Change a certain bit to 1
      //! \param index the bit index from most significant (0) to least significant
      void changeBitTo1(off_t index)
         {
            bitfield_t const ONE = 1;
            bitfield_t cleanerMask = ONE << (sizeof(bitfield_t)*8 - 1 - index);
            m_value = m_value | cleanerMask;
            m_mask = m_mask | cleanerMask;
         }
      
      //! \brief Change a certain bit to X
      //! \param index the bit index from most significant (0) to least significant
      void changeBitToX(off_t index)
         {
            bitfield_t const ONE = 1;
            bitfield_t cleanerMask = ONE << (sizeof(bitfield_t)*8 - 1 - index);
            m_value = m_value & ~cleanerMask;
            m_mask = m_mask & ~cleanerMask;
         }
      
      //! \brief Get the value of the most significant bit
      //! \returns the value of the most significant bit
      bit_value_t getFirstBit() const
         { return (*this)[0]; }
      
      //! \brief Get the position (not index) of the most significant bit
      //! \returns the position of the most significant bit
      int getFirstBitNumber() const
         { return m_firstBit; }
      
      //! \brief Get the number of low bits necessary to complete a word
      //! \returns the number of low bits necessary to complete a word
      int getBitsToEnd() const
         { return sizeof(bitfield_t) * 8 - m_firstBit - getLength(); }
      
      //! \brief Check if it contains the least significant bit
      //! \returns true if it contains the least significant bit
      bool containsLastBit() const
         { return m_firstBit + getLength() == sizeof(bitfield_t) * 8; }
      
      //! \brief Get the most significant bits that are common
      //! \param other the other region
      //! \param[out] prefix the common prefix
      void getCommonPrefix(Region const &other, /* Outputs: */ Region &prefix) const
         {
            bitfield_t const ONE = 1;
            
            int shortestLengthIndex = (m_nextBit > other.m_nextBit ? m_nextBit : other.m_nextBit);
            signed char firstMismatch = (sizeof(bitfield_t) * 8) - 1;
            bitfield_t cleanerMask = 0;
            cleanerMask--;
            while (
               firstMismatch > shortestLengthIndex
               &&
               ( m_mask & (ONE << firstMismatch) ) == ( other.m_mask & (ONE << firstMismatch) )
               &&
               ( m_value & (ONE << firstMismatch) ) == ( other.m_value & (ONE << firstMismatch) )
            ) {
               firstMismatch--;
               cleanerMask = cleanerMask >> 1;
            }
            
            prefix.m_firstBit = m_firstBit;
            prefix.m_nextBit = firstMismatch;
            prefix.m_mask = m_mask & ~cleanerMask;
            prefix.m_value = m_value & ~cleanerMask;
         }
      
      
      //! \brief Get the most significant bits that are common until this subsumes
      //! \param other the other region
      //! \param[out] prefix the common prefix
      //! \returns true if the following bit subsumes
      bool getPrefixUntilThisSubsumes(Region const &other, /* Outputs: */ Region &prefix) const
         {
            bitfield_t const ONE = 1;
            bool result = false;
            
            int shortestLengthIndex = (m_nextBit > other.m_nextBit ? m_nextBit : other.m_nextBit);
            signed char firstMismatch = sizeof(bitfield_t)*8 - 1;
            bitfield_t cleanerMask = 0;
            cleanerMask--;
            while (firstMismatch > shortestLengthIndex) {
               if (m_mask & (ONE << firstMismatch)) {
                  if (other.m_mask & (ONE << firstMismatch)) {
                     if ( (m_value & (ONE << firstMismatch)) != (other.m_value & (ONE << firstMismatch)) ) {
                        // The current bit is different
                        result = false;
                        break;
                     }
                  } else {
                     // other subsumes
                  }
               } else {
                  // This has an X
                  if (other.m_mask & (ONE << firstMismatch)) {
                     // This subsumes
                     result = true;
                     break;
                  }
               }
               
               firstMismatch--;
               cleanerMask = cleanerMask >> 1;
            }
            
            prefix.m_firstBit = m_firstBit;
            prefix.m_nextBit = firstMismatch;
            prefix.m_mask = m_mask & ~cleanerMask;
            prefix.m_value = m_value & ~cleanerMask;
            
            return result;
         }
      
      
      //! \brief Generate a region that extends this region to cover the extra elements of another without allowing aliasing
      //! \param other the other region
      //! \param[out] result the extended region
      //! \returns true if the combination is possible without aliasing
      inline bool combine(Region const &other, /* Outputs: */ Region &result) const;
      
      
      //! \brief Extends it to also cover another region independently of aliasing
      //! \param other the other region
      void operator|=(Region const &other)
         {
            // NOTE: m_firstBit == other.m_firstBit && m_nextBit == other.m_nextBit
            m_mask = m_mask & other.m_mask & ~(m_value ^ other.m_value);
            m_value = m_value & other.m_mask & ~(m_value ^ other.m_value);
         }
      
      
      //! \brief Find the intersection with another region
      //! \param other the other region
      //! \returns the intersection with other
      //! \pre this and other intersect
      Region intersect(Region const &other) const
         {
            // NOTE: m_firstBit == other.m_firstBit && m_nextBit == other.m_nextBit
            Region result(*this);
            result.m_mask |= other.m_mask;
            result.m_value |= other.m_value;
            result.m_value &= result.m_mask;
            return result;
         }
      
      
      //! \brief Checks if it intersects with another region
      //! \param other the other region
      //! \returns true if it intersects with other
      bool doIntersect(Region const &other) const
         {
            bitfield_t combinedMask = m_mask & other.m_mask;
            return (m_value & combinedMask) == (other.m_value & combinedMask);
         }
      
      
      //! \brief Number of elements/bytes that it covers
      //! \returns the number of elements/bytes covered
      size_t getBreadth() const
         {
            size_t result = 1;
            bitfield_t mask = m_mask;
            while (mask) {
               if (!(mask & 1)) {
                  result = result << 1;
               }
               mask = mask >> 1;
            }
            return result;
         }
      
      
      //! \brief Number of elements/bytes that it covers
      //! \returns the number of elements/bytes covered
      void extend(bitfield_t const &bitMask)
         {
            m_mask = m_mask & ~bitMask;
            m_value = m_value & ~bitMask;
         }
      
      //! \brief Lowest value covered
      //! \returns the lowest value covered
      bitfield_t getFirstValue() const
         { return m_value; }
      
      
      //! \brief Lowest value of the intersection with another region
      //! \param other the other region
      //! \returns the lowest value covered by the intersection
      //! \pre this and other intersect
      Region rectilinearMin(Region const &other) const
         {
            bitfield_t const ZERO = 0;
            Region result = *this;
            result.m_mask = ~ZERO;
            result.m_value = result.m_value & other.m_value;
            return result;
         }
      
      //! \brief Size in elements/bytes of the least significant extent/chunk
      //! \returns the size in elements/bytes of the least significant extent/chunk
      size_t getContiguousChunkLength() const
         {
            bitfield_t const ZERO = 0;
            bitfield_t const ONE = 1;
            size_t bitSelector = ONE;
            
            while (bitSelector != ZERO) {
               if (bitSelector & m_mask) {
                  return bitSelector;
               }
               bitSelector = bitSelector << 1;
            }
            
            return ~ZERO; // Actually this is an overflow since the real value is the same + 1
         }
      
      /*! \brief Clones the dependency object.
       */
      BaseDependency* clone() const;
      
      virtual bool overlap ( const BaseDependency &obj ) const;
      
      friend class RegionPart;
      friend std::ostream& operator<< (std::ostream& o, Region const &region);

      //! \brief Returns dependency base address
      virtual void * getAddress () const;
   };


} // namespace nanos


#endif // _NANOS_REGION_DECL
