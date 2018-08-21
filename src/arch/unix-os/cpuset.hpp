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

#ifndef CPUSET_HPP
#define CPUSET_HPP

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <string.h>
#include <ostream>
#include "compatibility.hpp"

namespace nanos {

class CpuSet
{
   public:
      class CpuSetConstIterator
      {
         private:
            const CpuSet &_cpuset;
            size_t _pos;
         public:
            CpuSetConstIterator( const CpuSet &cpuset, size_t pos )
               : _cpuset(cpuset), _pos(pos) {}
            CpuSetConstIterator( const CpuSetConstIterator& it )
               : _cpuset(it._cpuset), _pos(it._pos) {}
            ~CpuSetConstIterator() {}

            size_t operator*() const { return _pos; }
            CpuSetConstIterator& operator++() { forward(); return *this; }
            CpuSetConstIterator& operator--() { backward(); return *this; }

            friend bool operator==(
                  const CpuSetConstIterator& lhs, const CpuSetConstIterator& rhs );
            friend bool operator!=(
                  const CpuSetConstIterator& lhs, const CpuSetConstIterator& rhs );

         private:
            void forward();
            void backward();
      };

      typedef CpuSetConstIterator const_iterator;

   private:
      cpu_set_t _mask;
   public:
      // Default constructor
      CpuSet(): _mask() {}

      CpuSet( int cpuid ): _mask() { set(cpuid); }

      // Destructor
      ~CpuSet() {}

      // Copy constructors
      CpuSet( const cpu_set_t* cpu_set )
      {
         ::memcpy( &_mask, cpu_set, sizeof(cpu_set_t));
      }

      CpuSet( const cpu_set_t& cpu_set )
      {
         ::memcpy( &_mask, &cpu_set, sizeof(cpu_set_t));
      }

      CpuSet( const CpuSet& cpu_set )
      {
         ::memcpy( &_mask, &(cpu_set._mask), sizeof(cpu_set_t));
      }

      // Assignment operators
      CpuSet& operator=( const cpu_set_t& rhs )
      {
         ::memcpy( &_mask, &rhs, sizeof(cpu_set_t));
         return *this;
      }

      CpuSet& operator=( const CpuSet& rhs )
      {
         ::memcpy( &_mask, &(rhs._mask), sizeof(cpu_set_t));
         return *this;
      }

      // Arithmetic and logical operators.
      // All of them are friend methods because both operands are const
      friend CpuSet operator|( const CpuSet& lhs, const CpuSet& rhs );
      friend CpuSet operator&( const CpuSet& lhs, const CpuSet& rhs );
      friend CpuSet operator+( const CpuSet& lhs, const CpuSet& rhs );
      friend CpuSet operator*( const CpuSet& lhs, const CpuSet& rhs );
      friend CpuSet operator-( const CpuSet& lhs, const CpuSet& rhs );
      friend bool operator==( const CpuSet& lhs, const CpuSet& rhs );
      friend bool operator!=( const CpuSet& lhs, const CpuSet& rhs );

      CpuSet& operator|=( const CpuSet& rhs )
      {
         this->add( rhs );
         return *this;
      }

      CpuSet& operator&=( const CpuSet& rhs )
      {
         this->multiply( rhs );
         return *this;
      }

      CpuSet& operator+=( const CpuSet& rhs )
      {
         this->add( rhs );
         return *this;
      }

      CpuSet& operator*=( const CpuSet& rhs )
      {
         this->multiply( rhs );
         return *this;
      }

      void copyTo( cpu_set_t *cpu_set ) const
      {
         ::memcpy( cpu_set, &_mask, sizeof(cpu_set_t));
      }

      size_t size() const
      {
         return CPU_COUNT( &_mask );
      }

      bool isSet(int n) const
      {
         return CPU_ISSET( n, &_mask );
      }

      void set(int n)
      {
         CPU_SET( n, &_mask );
      }

      void clear(int n)
      {
         CPU_CLR( n, &_mask );
      }

      void add ( const cpu_set_t& cpu_set )
      {
         CPU_OR( &_mask, &_mask, &cpu_set );
      }

      void add ( const CpuSet& cpu_set )
      {
         CPU_OR( &_mask, &_mask, &(cpu_set._mask) );
      }

      void multiply ( const cpu_set_t& cpu_set )
      {
         CPU_AND( &_mask, &_mask, &cpu_set );
      }

      void multiply ( const CpuSet& cpu_set )
      {
         CPU_AND( &_mask, &_mask, &(cpu_set._mask) );
      }

      bool isSubsetOf ( const CpuSet& cpu_set ) const
      {
         cpu_set_t intxn;
         CPU_AND( &intxn, &_mask, &(cpu_set._mask) );
         return CPU_EQUAL( &intxn, &_mask );
      }

      bool isSupersetOf ( const CpuSet& cpu_set ) const
      {
         cpu_set_t intxn;
         CPU_AND( &intxn, &_mask, &(cpu_set._mask) );
         return CPU_EQUAL( &intxn, &(cpu_set._mask) );
      }

      // low level
      cpu_set_t& get_cpu_set() { return _mask; }
      const cpu_set_t& get_cpu_set() const { return _mask; }

      cpu_set_t* get_cpu_set_pointer() { return &_mask; }
      const cpu_set_t* get_cpu_set_pointer() const { return &_mask; }

      // verbose methods
      std::string toString() const;

      // iterator
      size_t first() const;
      size_t last() const;
      const_iterator begin() const { return const_iterator( *this, first() ); }
      const_iterator end() const { return const_iterator( *this, last() ); }
};


// Non-member functions

inline CpuSet operator|( const CpuSet& lhs, const CpuSet& rhs )
{
   CpuSet result;
   CPU_OR( &result._mask, &lhs._mask, &rhs._mask );
   return result;
}

inline CpuSet operator&( const CpuSet& lhs, const CpuSet& rhs )
{
   CpuSet result;
   CPU_AND( &result._mask, &lhs._mask, &rhs._mask );
   return result;
}

inline CpuSet operator+( const CpuSet& lhs, const CpuSet& rhs )
{
   return lhs | rhs;
}

inline CpuSet operator*( const CpuSet& lhs, const CpuSet& rhs )
{
   return lhs & rhs;
}

inline CpuSet operator-( const CpuSet& lhs, const CpuSet& rhs )
{
   CpuSet result;
   CPU_XOR( &result._mask, &lhs._mask, &rhs._mask );
   CPU_AND( &result._mask, &lhs._mask, &result._mask );
   return result;
}

inline bool operator==( const CpuSet& lhs, const CpuSet& rhs )
{
   return CPU_EQUAL( &lhs._mask, &rhs._mask );
}

inline bool operator!=( const CpuSet& lhs, const CpuSet& rhs )
{
   return !( rhs == lhs );
}

inline std::ostream& operator<<(std::ostream& os, const CpuSet& cpu_set)
{
   os << cpu_set.toString();
   return os;
}

inline bool operator==( const CpuSet::const_iterator& lhs, const CpuSet::const_iterator& rhs )
{
   return (lhs._cpuset == rhs._cpuset) && (lhs._pos == rhs._pos);
}

inline bool operator!=( const CpuSet::const_iterator& lhs, const CpuSet::const_iterator& rhs )
{
   return !( rhs == lhs );
}

} // namespace nanos

#endif /* CPUSET_HPP */
