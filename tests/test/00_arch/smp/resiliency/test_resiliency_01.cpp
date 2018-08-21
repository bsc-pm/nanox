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
 test_generator='gens/resiliency-generator'
 </testinfo>
 */

#include <iostream>
#include "system.hpp"
#include "smpprocessor.hpp"
#include "basethread.hpp"
#include "debug.hpp"

using namespace std;

typedef struct
{
      int depth;
      int executions_so_far;
} arg_dig_t;

int cutoff_value = 10;
bool errors = false;

void
set_true ( void* );

void
fail ( bool );

void
dig_0 ( void* );

void
dig ( int );

void set_true ( void *ptr )
{
   bool* arg = (bool*) ptr;
   *arg = true;
}

// Tries to write to an invalid address: SIGSEGV
void fail ( bool flag )
{
   if (flag) {
      int *a;
      a = 0;
      *a = 1;
   }
}

void dig_0 ( void *ptr )
{
   arg_dig_t * args = (arg_dig_t *) ptr;
   args->executions_so_far++;

   if (args->depth >= cutoff_value) {
      fail(true);
   } else {
      dig(args->depth);
   }
}

void dig ( int d )
{
   nanos::WD *this_wd = nanos::getMyThreadSafe()->getCurrentWD();

   arg_dig_t * args1 = new arg_dig_t();
   args1->depth = d + 1;
   args1->executions_so_far = 0;

   nanos::WD * wd1 = new nanos::WD(new nanos::ext::SMPDD(dig_0),
         sizeof(arg_dig_t), __alignof__(arg_dig_t), args1);
   wd1->setRecoverable(false);

   this_wd->addWork(*wd1);
   if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
      char *idata = NEW char[sys.getPMInterface().getInternalDataSize()];
      sys.getPMInterface().initInternalData( idata );
      wd1->setInternalData( idata );
   }

   sys.setupWD(*wd1, this_wd);
   nanos::sys.submit(*wd1);

//	#pragma omp taskwait
   this_wd->waitCompletion();

   if (wd1->isInvalid()) {
//      try to submit another task and check if it executes (should not, even if it's recoverable)
      bool executed = false;

      nanos::WD * wd = new nanos::WD(new nanos::ext::SMPDD(set_true),
            sizeof(bool*), __alignof__(bool*), &executed);
      wd->setRecoverable(true);
      this_wd->addWork(*wd);
      if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
         char *idata = NEW char[sys.getPMInterface().getInternalDataSize()];
         sys.getPMInterface().initInternalData( idata );
         wd->setInternalData( idata );
      }

      sys.setupWD(*wd, this_wd);
      nanos::sys.submit(*wd);
      this_wd->waitCompletion();
//      check
      if (executed) {
         errors = true;
         cerr
               << "Error: tasks invalidated before their execution should not be executed."
               << endl;
      }
   } else // !invalid
   {
      errors = true;
      cerr << "Error: whenever no recoverable tasks exist, " << endl;
   }
}

int main ( int argc, char **argv )
{
   if (argc > 1)
      cutoff_value = atoi(argv[1]);
   nanos::WD* this_wd = getMyThreadSafe()->getCurrentWD();
   this_wd->setRecoverable(true);
   dig(0);
   this_wd->setInvalid(false);
   if (errors)
      return 1;
   return 0;
}
