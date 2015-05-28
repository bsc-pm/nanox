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

#ifndef CPUSET_DECL_HPP
#define CPUSET_DECL_HPP

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <string.h>
#include "compatibility.hpp"

class CpuSet
{
   private:
      cpu_set_t _mask;
   public:
      CpuSet()
      {
         CPU_ZERO( &_mask );
      }

      CpuSet( const cpu_set_t* set )
      {
         ::memcpy( &_mask, set, sizeof(cpu_set_t));
      }

      CpuSet( const cpu_set_t& set )
      {
         ::memcpy( &_mask, &set, sizeof(cpu_set_t));
      }

      CpuSet( const CpuSet& set )
      {
         ::memcpy( &_mask, &(set._mask), sizeof(cpu_set_t));
      }

      CpuSet& operator=( const cpu_set_t& set )
      {
         ::memcpy( &_mask, &set, sizeof(cpu_set_t));
         return *this;
      }

      CpuSet& operator=( const CpuSet& set )
      {
         ::memcpy( &_mask, &(set._mask), sizeof(cpu_set_t));
         return *this;
      }

      friend CpuSet operator|( const CpuSet& set1, const CpuSet& set2 );
      friend CpuSet operator&( const CpuSet& set1, const CpuSet& set2 );
      friend CpuSet operator+( const CpuSet& set1, const CpuSet& set2 );
      friend CpuSet operator*( const CpuSet& set1, const CpuSet& set2 );
      friend bool operator==( const CpuSet& set1, const CpuSet& set2 );
      friend bool operator!=( const CpuSet& set1, const CpuSet& set2 );

      void copyTo( cpu_set_t *set ) const
      {
         ::memcpy( set, &_mask, sizeof(cpu_set_t));
      }

      size_t size() const
      {
         return CPU_COUNT( &_mask );
      }

      size_t countCommon( const CpuSet& set ) const
      {
         cpu_set_t mask;
         CPU_AND( &mask, &_mask, &set._mask );
         return CPU_COUNT( &mask );
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

      void add ( const cpu_set_t& set )
      {
         CPU_OR( &_mask, &_mask, &set );
      }

      void add ( const CpuSet& set )
      {
         CPU_OR( &_mask, &_mask, &(set._mask) );
      }

      // low level
      cpu_set_t& get_cpu_set() { return _mask; }
      const cpu_set_t& get_cpu_set() const { return _mask; }
};

inline CpuSet operator|( const CpuSet& set1, const CpuSet& set2 )
{
   CpuSet result;
   CPU_OR( &result._mask, &set1._mask, &set2._mask );
   return result;
}

inline CpuSet operator&( const CpuSet& set1, const CpuSet& set2 )
{
   CpuSet result;
   CPU_AND( &result._mask, &set1._mask, &set2._mask );
   return result;
}

inline CpuSet operator+( const CpuSet& set1, const CpuSet& set2 )
{
   return set1 | set2;
}

inline CpuSet operator*( const CpuSet& set1, const CpuSet& set2 )
{
   return set1 & set2;
}

inline bool operator==( const CpuSet& set1, const CpuSet& set2 )
{
   return CPU_EQUAL( &set1._mask, &set2._mask );
}

inline bool operator!=( const CpuSet& set1, const CpuSet& set2 )
{
   return !( set2 == set1 );
}

#endif /* CPUSET_DECL_HPP */
