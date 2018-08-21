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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <sstream>
#include "os.hpp"
#include "cpuset.hpp"

using namespace nanos;

/*
 * Returns human readable representation of the cpuset. The output format is
 * a list of CPUs with ranges (for example, "0,1,3-9").
 */
std::string CpuSet::toString() const
{
   std::ostringstream s;

   bool entry_made = false;
   size_t max = OS::getMaxProcessors();
   for ( size_t i=0; i<max; ++i ) {
      if ( isSet(i) ) {

         // Find range size
         size_t run = 0;
         for ( size_t j=i+1; j<max; ++j ) {
            if ( isSet(j) ) ++run;
            else break;
         }

         // Add ',' separator for subsequent entries
         if ( entry_made ) {
            s << ",";
         } else {
            entry_made = true;
         }

         // Write element, pair or range
         if ( run == 0 ) {
            s << i;
         } else if ( run == 1 ) {
            s << i << "," << i+1;
            ++i;
         } else {
            s << i << "-" << i+run;
            i+=run;
         }
      }
   }

   return s.str();
}

size_t CpuSet::first() const
{
   size_t first_cpu = 0;
   if ( size() > 0 ) {
      while ( !isSet(first_cpu) ) { ++first_cpu; }
   }
   return first_cpu;
}

size_t CpuSet::last() const
{
   size_t last_cpu = 0;
   size_t remaining = size();
   size_t max = OS::getMaxProcessors();
   for ( size_t i=0; i<max && remaining>0; ++i ) {
      if ( isSet(i) ) {
         last_cpu = i+1;
         --remaining;
      }
   }
   return last_cpu;
}

void CpuSet::const_iterator::forward()
{
   // Do not go forward if _pos is not pointing to a set bit
   if ( !_cpuset.isSet(_pos) ) return;

   size_t i;
   size_t max = OS::getMaxProcessors();
   for ( i=_pos+1; i<max; ++i ) {
      if ( _cpuset.isSet(i) ) {
         break;
      }
   }
   // Update if i is valid, otherwise last+1
   _pos = (i<max) ? i : _pos + 1;
}

void CpuSet::const_iterator::backward()
{
   size_t i = _pos;
   do {
      --i;
      if ( _cpuset.isSet(i) ) {
         _pos = i;
         break;
      }
   } while (i!=0);
}
