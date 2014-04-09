/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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
 test_generator="gens/resiliency-generator"
 </testinfo>
 */

#include "config.hpp"
#include "nanos.h"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"
#include <string.h>
#include <signal.h>

using namespace std;

bool errors = false;

typedef struct
{
  int signo;
} arg_t;

void
testSignal(void*);
void
testSignals(void*);

void
testSignal(void *arg)
{
  arg_t* func_arg = (arg_t*) arg;
  raise(func_arg->signo);
}

void
testSignals(void *arg)
{
  int num_signals = 5;
  arg_t signals[5] =
    {
      { SIGILL },
      { SIGTRAP },
      { SIGBUS },
      { SIGSEGV },
      { SIGFPE } };

  nanos::WD* this_wd = getMyThreadSafe()->getCurrentWD();

  WD* tasks[5];
  for (int i = 0; i < num_signals; i++)
    {
      tasks[i] = new nanos::WD(new nanos::ext::SMPDD(testSignal), sizeof(arg_t),
          __alignof__(arg_t), &signals[i]);

      this_wd->addWork(*tasks[i]);
      sys.setupWD(*tasks[i], this_wd);
      sys.submit(*tasks[i]);
    }

  this_wd->waitCompletion();

  if (!this_wd->isInvalid())
    {
      errors = true;
      cerr
          << "Error: Non-recoverable children detected an error but this task wasn't invalidated."
          << endl;
    }
  else if (this_wd->getDepth() == 0)
    {
      this_wd->setInvalid(false); // Unset invalid bit to avoid fatal error.
      this_wd->getParent()->setInvalid(false);
    }

  for (int i = 0; i < num_signals && !errors; i++)
    {
      if (!tasks[i]->isInvalid())
        {
          errors = true;
          cerr << "Error: handled " << strsignal(signals[i].signo)
              << " but the task wasn't invalidated." << endl;
        }
    }
}

int
main(int argc, char **argv)
{
  cout << "start" << endl;

  nanos::WD* this_wd = getMyThreadSafe()->getCurrentWD();

  WD* task = new nanos::WD(new nanos::ext::SMPDD(testSignals));

  this_wd->addWork(*task);
  sys.submit(*task);

  this_wd->waitCompletion();

  if (errors)
    {
      cout << "end: errors detected" << endl;
      return 1;
    }

  cout << "end: success" << endl;
  return 0;
}
