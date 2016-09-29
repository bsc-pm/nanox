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

/*
<testinfo>
   test_generator="gens/core-generator"
   test_generator_ENV=( "NX_TEST_MODE=performance"
                        "NX_TEST_MAX_CPUS=1"
                        "NX_TEST_SCHEDULE=bf"
                        "NX_TEST_ARCH=smp" )
</testinfo>
*/

#include <cstdlib>
#include <assert.h>
#include "cpuset.hpp"

using namespace nanos;

int main(int argc, char *argv[])
{
   CpuSet set1;
   assert(set1.size()==0);

   set1.set(0);
   set1.set(1);   /* 0011 */
   assert(set1.size()==2);

   // Copy ctor and logical operators
   CpuSet set2(set1);
   assert(set1.size()==2);
   assert(set1 == set2);
   set1.set(2);   /* 0111 */
   assert(set1 != set2);

   // Arithmetic operators
   set2.set(3);
   set2.clear(1); /* 1001 */
   assert((set1 | set2).size()==4);    /* 1111 */
   assert((set1 + set2).size()==4);    /* 1111 */
   assert((set1 & set2).size()==1);    /* 0001 */
   assert((set1 * set2).size()==1);    /* 0001 */

   // Compound assignment operators
   set2 = set1;   /* 0111 */
   set1.set(3);   /* 1111 */
   set2 &= set1;  /* 0111 */
   assert(set2.size()==3);
   set2 |= set1;  /* 1111 */
   assert(set2.size()==4);

   // First and last methods
   CpuSet set3;   /* 0000 */
   assert(set3.first()==0);
   assert(set3.last()==0);
   set3.set(0);   /* 0001 */
   assert(set3.first()==0);
   assert(set3.last()==1);
   set3.set(1);   /* 0011 */
   assert(set3.first()==0);
   assert(set3.last()==2);
   set3.set(3);   /* 1011 */
   assert(set3.first()==0);
   assert(set3.last()==4);
   set3.clear(0); /* 1010 */
   assert(set3.first()==1);
   assert(set3.last()==4);

   // Iterators
   CpuSet::const_iterator it = set3.begin();
   assert( *it == 1 );
   ++it;
   assert( *it == 3 );
   ++it;
   assert( it == set3.end() );
   ++it;
   assert( it == set3.end() );
   --it;
   assert( *it == 3 );
   --it;
   assert( *it == 1 );
   assert( it == set3.begin() );
   --it;
   assert( *it == 1 );
   assert( it == set3.begin() );

   return EXIT_SUCCESS;
}
