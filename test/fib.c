#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <nanos.h>

int cutoff_value = 10;

int fib_seq (int n)
{
	int x, y;
	if (n < 2) return n;

	x = fib_seq(n-1);
	y = fib_seq(n-2);

	return x + y;
}

int fib (int n, int d);

typedef struct {
   int n;
   int d;
   int *x;
} fib_args;

void fib_1(void *ptr)
{
   fib_args * args = (fib_args *)ptr;
  *args->x = fib(args->n-1,args->d+1);
}

void fib_2(void *ptr)
{
   fib_args * args = (fib_args *)ptr;
   *args->x = fib(args->n-2,args->d+1);
}

nanos_smp_args_t fib_device_arg_1 = { fib_1 };
nanos_smp_args_t fib_device_arg_2 = { fib_2 };

nanos_device_t fib_devices_1[] =
{ {nanos_smp_factory, &fib_device_arg_1 } };

nanos_device_t fib_devices_2[] =
{ {nanos_smp_factory, &fib_device_arg_1 } };

int fib (int n, int d)
{
	int x, y;
	if (n < 2) return n;

	if ( d < cutoff_value ) {
//       #pragma omp task untied shared(x) firstprivate(n,d)
//      x = fib(n - 1,d+1);
       {
       nanos_wd_t wd=0;
       fib_args *args=0;

      nanos_wd_props_t props;
      props.mandatory_creation=true;

       nanos_create_wd ( &wd, 1, fib_devices_1 , sizeof(fib_args),
                         (void **)&args, nanos_current_wd(), &props );
       args->n = n; args->d = d; args->x = &x;
       nanos_submit(wd,0,0);
       }

//		#pragma omp task untied shared(y) firstprivate(n,d)
//		y = fib(n - 2,d+1);
       {
       nanos_wd_t wd=0;
       fib_args *args=0;

      nanos_wd_props_t props;
      props.mandatory_creation=true;


       nanos_create_wd ( &wd, 1, fib_devices_2 , sizeof(fib_args),
                         (void **)&args, nanos_current_wd(), &props );
       args->n = n; args->d = d; args->x = &y;
       nanos_submit(wd,0,0);
       }
//		#pragma omp taskwait
	   nanos_wg_wait_completation(nanos_current_wd());
	} else {
		x = fib_seq(n-1);
		y = fib_seq(n-2);
	}

	return x + y;
}

double get_wtime(void)
{
        struct timeval ts;
        double t;
        int err;

        err = gettimeofday(&ts, NULL);
        t = (double) (ts.tv_sec)  + (double) ts.tv_usec * 1.0e-6;

        return t;
}

void fib0 (int n)
{
	double start,end;
	int par_res;

	start = get_wtime();
	par_res = fib(n,0);
	end = get_wtime();

    printf("Fibonacci result for %d is %d\n", n, par_res);
	printf("Computation time: %f seconds.\n",  end - start);
}


int main (int argc, char **argv ) 
{
	int n=25;
	if (argc > 1) n = atoi(argv[1]);
	fib0(n);
    return 0;
}
