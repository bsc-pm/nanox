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

/*
<testinfo>
test_generator=gens/core-generator
test_generator_ENV=( "NX_TEST_MAX_CPUS=1"
                     "NX_TEST_SCHEDULE=bf" )
</testinfo>
*/

#include "system.hpp"
#include <iostream>

using namespace std;
using namespace nanos;

#define SIZEOF_WD             256*sizeof(void *)
#define SIZEOF_DOWAIT          40*sizeof(void *)
#define SIZEOF_DOSUBMIT        32*sizeof(void *)
#define SIZEOF_ICONTEXT        32*sizeof(void *)

int main ( int argc, char **argv )
{
   int error = 0;

   cout << "Size of WorkDescriptor is " << sizeof(WD) << " out of " << SIZEOF_WD << endl;
   if ( sizeof(WD) > SIZEOF_WD ) error = 1;

   cout << "Size of DOWait is " << sizeof(DOWait) << " out of " << SIZEOF_DOWAIT << endl;
   if ( sizeof(DOWait) > SIZEOF_DOWAIT ) error = 1;

   cout << "Size of LazyInit<DOWait> is " << sizeof(LazyInit<DOWait>) << " out of " << SIZEOF_DOWAIT << endl;
   if ( sizeof(LazyInit<DOWait>) > SIZEOF_DOWAIT ) error = 1;

   cout << "Size of DOSubmit is " << sizeof(DOSubmit) << " out of " << SIZEOF_DOSUBMIT << endl;
   if ( sizeof(DOSubmit) > SIZEOF_DOSUBMIT ) error = 1;

   cout << "Size of LazyInit<DOSubmit> is " << sizeof(LazyInit<DOSubmit>) << " out of " << SIZEOF_DOSUBMIT << endl;
   if ( sizeof(LazyInit<DOSubmit>) > SIZEOF_DOSUBMIT ) error = 1;

   cout << "Size of InstrumentationContextData is " << sizeof(InstrumentationContextData) << " out of " << SIZEOF_ICONTEXT << endl;
   if ( sizeof(InstrumentationContextData) > SIZEOF_ICONTEXT ) error = 1;

   return error;
}
