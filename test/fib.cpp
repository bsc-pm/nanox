#include <iostream>
#include "system.hpp"
#include "smpprocessor.hpp"
#include "smpwd.hpp"

int cutoff_value = 10;

int fib_seq (int n)
{
	int x, y;
	if (n < 2) return n;

	x = fib_seq(n-1);
	y = fib_seq(n-2);

	return x + y;
}

int fib (int n, int d, nanos::WD* parent);

void fib_0(nanos::WD *wd)
{
int n = wd->getValue<int>(0);
int d = wd->getValue<int>(1);
int *x = wd->getValue<int *>(2);
nanos::WD * p = wd->getValue<nanos::WD *>(3);

*x = fib(n-1,d+1,p);
}

void fib_1(nanos::WD *wd)
{
int n = wd->getValue<int>(0);
int d = wd->getValue<int>(1);
int *x = wd->getValue<int *>(2);
nanos::WD * p = wd->getValue<nanos::WD *>(3);

*x = fib(n-2,d+1,p);
}


int fib (int n, int d, nanos::WD* parent)
{
  int x, y;
  if (n < 2) return n;
  
  if ( d < cutoff_value ) {
    nanos::WG *wg = new nanos::WG();

    //		#pragma omp task untied shared(x) firstprivate(n,d)
    //		x = fib(n - 1,d+1);
    {
      nanos::WorkData *data = new nanos::WorkData();
      data->setArguments(2*sizeof(int)+sizeof(int *)+sizeof(nanos::WD *),4,0,n,d,&x,parent);
      nanos::SMPWD * wd = new nanos::SMPWD(fib_0,data);
      wg->addWork(*wd); 
      nanos::sys.submit(*wd, parent);
    }
    
    //		#pragma omp task untied shared(y) firstprivate(n,d)
    //		y = fib(n - 2,d+1);
    {
      nanos::WorkData *data = new nanos::WorkData();
      data->setArguments(2*sizeof(int)+sizeof(int *)+sizeof(nanos::WD *),4,0,n,d,&y,parent);
      nanos::SMPWD * wd = new nanos::SMPWD(fib_1,data);
      wg->addWork(*wd);
      nanos::sys.submit(*wd, parent);
    }
    
    //		#pragma omp taskwait
    wg->waitCompletation();
  } else {
    x = fib_seq(n-1);
    y = fib_seq(n-2);
  }
  
  return x + y;
}


void fib0 (int n)
{
	int par_res = fib(n,0,NULL);

	std::cout << "Fibonacci result for " << n << " is " << par_res << std::endl;
}


int main (int argc, char **argv ) 
{
	int n=25;
	if (argc > 1) n = atoi(argv[1]);
	fib0(n);
}
