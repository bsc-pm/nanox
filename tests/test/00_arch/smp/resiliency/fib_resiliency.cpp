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
 test_generator='gens/resiliency-generator -a --task-retries=4'
 </testinfo>
 */

#include <iostream>
#include "system.hpp"
#include "smpprocessor.hpp"
#include "basethread.hpp"
#include <sys/time.h>

using namespace std;

bool errors = false;

void
fail();
int
fib_seq(int n);
void
fib_0(void *ptr);
void
fib_1(void *ptr);
double
get_wtime(void);
int
fib0(int n);

int cutoff_value = 5;

// Try to write to an invalid address: SIGSEGV
void
fail()
{
  int *a;
  a = 0;
  *a = 1;
}

int
fib_seq(int n)
{
  int x, y;

  if (n < 2)
    return n;

  x = fib_seq(n - 1);

  y = fib_seq(n - 2);

  return x + y;
}

int
fib(int n, int d);

typedef struct
{
  int n; // Arg
  int d; // Depth
  int *x; // Result
  bool must_fail; // Force an error?
  int executions_so_far;
} fib_args;

void
fib_0(void *ptr)
{
  fib_args * args = (fib_args *) ptr;
  args->executions_so_far++;

  if (args->must_fail)
    {
      if (args->executions_so_far < 3)
        fail();
      else
        args->must_fail = false;
    }

  *args->x = fib(args->n - 1, args->d + 1);
}

void
fib_1(void *ptr)
{
  fib_args * args = (fib_args *) ptr;
  args->executions_so_far++;

  *args->x = fib(args->n - 2, args->d + 1);
}

int
fib(int n, int d)
{
  int x, y;

  if (n < 2)
    return n;

  nanos::WD *wg = nanos::getMyThreadSafe()->getCurrentWD();

//	#pragma omp task untied shared(x) firstprivate(n,d) recover
//	x = fib(n - 1,d+1);
  fib_args * args1 = new fib_args();
  args1->n = n;
  args1->d = d;
  args1->x = &x;
  args1->must_fail = true;
  args1->executions_so_far = 0;

  nanos::WD * wd1 = new nanos::WD(new nanos::ext::SMPDD(fib_0),
      sizeof(fib_args), __alignof__(fib_args), args1);
  wd1->setRecoverable(true);
  wg->addWork(*wd1);
  nanos::sys.submit(*wd1);

//	#pragma omp task untied shared(y) firstprivate(n,d)
//	y = fib(n - 2,d+1);
  fib_args * args2 = new fib_args();
  args2->n = n;
  args2->d = d;
  args2->x = &y;
  args2->executions_so_far = 0;

  nanos::WD * wd2 = new nanos::WD(new nanos::ext::SMPDD(fib_1),
      sizeof(fib_args), __alignof__(fib_args), args2);
  wg->addWork(*wd2);
  nanos::sys.submit(*wd2);

//	#pragma omp taskwait
  wg->waitCompletion();

//      check
  if (wd1->isInvalid() || wd2->isInvalid())
      errors = true;
  
  if (wg->getDepth() == 0 && wg->isInvalid())
      errors = true;
  
  return x + y;
}

double
get_wtime(void)
{

  struct timeval ts;
  double t;

  gettimeofday(&ts, NULL);
  t = (double) (ts.tv_sec) + (double) ts.tv_usec * 1.0e-6;

  return t;
}

int
fib0(int n)
{
  double start, end;
  int par_res;

  start = get_wtime();
  par_res = fib(n, 0);
  end = get_wtime();

  std::cout << "Fibonacci result for " << n << " is " << par_res << std::endl;
  std::cout << "Computation time:  " << end - start << " seconds." << std::endl;
  return par_res;
}

int
main(int argc, char **argv)
{
  int n = 20;

  if (fib0(n) != 6765)
    return 1;
  return 0;
}
