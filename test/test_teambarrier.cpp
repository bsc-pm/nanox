#include "config.hpp"
#include "nanos.h"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"
#include <string.h>

using namespace std;
using namespace nanos;
using namespace nanos::ext;

#define BARR_NUM 10

/*! works only with bf scheduler */
void barrier_code ( void * )
{
       for ( int i = 0; i < BARR_NUM; i++ ) {
              cout << "Before the barrier" << endl;
              nanos_team_barrier();
              cout << "After the barrier" << endl;

       }
}

int main (int argc, char **argv)
{
       cout << "start" << endl;
       //all threads perform a barrier: 
       for ( int i = 1; i < sys.getNumPEs(); i++ ) {
              WD * wd = new WD(new SMPDD(barrier_code));
              sys.submit(*wd);
       }
       usleep(100);

       barrier_code(NULL);

       cout << "end" << endl;
}
