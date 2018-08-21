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
 test_generator="gens/resiliency-generator"
 </testinfo>
 */

#include <iostream>
#include <string.h>
#include <sys/mman.h>
#include "config.hpp"
#include "smpdd.hpp"
#include "smpprocessor.hpp"
#include "system.hpp"

#include <errno.h>

using namespace std;

bool errors = false;
volatile bool wait = true;/* volatile means that this variable value can be modified 
                           * even if the execution flow of this program doesn't seem to do so.
                           */

void testSignal ( void* );
void testSignals ( void* );

void testSignal ( void *arg )
{
   int *array = (int*) arg;
   // First try
   for(int i = 0; i < 64; i++)
      array[i] += i; // Read & write into a memory space
   // Wait for another thread to (maybe) invalidate array
   while(wait){}
   // Second try
   for(int i = 0; i < 64; i++)
      array[i] += i; // Read & write into a memory space
}

void testSignals ( void *arg )
{
   nanos::WD* this_wd = getMyThreadSafe()->getCurrentWD();

   int *array = (int*) mmap(NULL, 64*sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED| MAP_ANONYMOUS, -1 , 0);
   if (array == MAP_FAILED)
      cerr << "Mmap failed: " << strerror(errno) << endl;
   
   // First phase: do a first dry run. This one should not fail.
   WD *task = new nanos::WD(new nanos::ext::SMPDD(testSignal), sizeof(int*),
         __alignof__(int*), array);
 
   this_wd->addWork(*task);
   if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
      char *idata = NEW char[sys.getPMInterface().getInternalDataSize()];
      sys.getPMInterface().initInternalData( idata );
      task->setInternalData( idata );
   }

   sys.setupWD(*task, this_wd);
   sys.submit(*task);

   /*
    * Although task serialization might seem to be incorrect is desired in this case.
    * This task is going to be invalidated when the signal is raised.
    * If other task happens to start its execution it will be skipped due to this invalidation.
    * This test just checks that the application does not crash due to the signal.
    */
   wait = false;
   __sync_synchronize();/* memory barrier (needed to make this change visible 
                         * for all threads in  ibm like power memory consistency model)
                         */

   this_wd->waitCompletion();
 
   if (task->isInvalid()) {
      errors = true;
      cerr
            << "Error: Unexpected task failure."
            << endl;
      this_wd->setInvalid(false);// In case this wd was invalidated
   }

   // Second phase: same as the previous but r/w protect the memory region.
   wait = true;
   __sync_synchronize();/* memory barrier (needed to make this change visible 
                         * for all threads in  ibm like power memory consistency model)
                         */
   WD *task2 = new nanos::WD(new nanos::ext::SMPDD(testSignal), sizeof(int*),
         __alignof__(int*), array);
 
   this_wd->addWork(*task2);
   if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
      char *idata = NEW char[sys.getPMInterface().getInternalDataSize()];
      sys.getPMInterface().initInternalData( idata );
      task2->setInternalData( idata );
   }

   sys.setupWD(*task2, this_wd);
   sys.submit(*task2);

   mprotect(array, 64*sizeof(int), PROT_NONE);
   wait = false;
   __sync_synchronize();/* memory barrier (needed to make this change visible 
                         * for all threads in  ibm like power memory consistency model)
                         */

   this_wd->waitCompletion();

   if (!this_wd->isInvalid()) {
      errors = true;
      cerr
            << "Error: Non-recoverable child detected an error but this task was not invalidated."
            << endl;
   } else {
      /*
       *  Revalidate this WD. It was invalidated by the tasks[i] when the signal was raised.
       *  This is expected behavior so, although revalidation must be only called by the runtime
       *  and not by the user, this is OK for testing.
        */
       this_wd->setInvalid(false);
       if (!task2->isInvalid()) {
          errors = true;
          cerr << "Error: seems that the signal was handled but the child task wasn't invalidated." << endl;
       }
   }
   this_wd->setInvalid(false);
}

int main ( int argc, char **argv )
{
   cout << "start" << endl;

   nanos::WD* this_wd = getMyThreadSafe()->getCurrentWD();

   WD* task = new nanos::WD(new nanos::ext::SMPDD(testSignals));
   task->setRecoverable(true);
   this_wd->addWork(*task);
   if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
      char *idata = NEW char[sys.getPMInterface().getInternalDataSize()];
      sys.getPMInterface().initInternalData( idata );
      task->setInternalData( idata );
   }

   sys.setupWD(*task, this_wd);
   sys.submit(*task);

   this_wd->waitCompletion();

   if (errors) {
      cout << "end: errors detected" << endl;
      return -1;
   }

   cout << "end: success" << endl;
   return 0;
}
