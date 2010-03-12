/*
<testinfo>
test_generator=gens/mixed-generator
</testinfo>
*/

#include "config.hpp"
#include "nanos.h"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"
#include "threadteam.hpp"
#include <string.h>

using namespace std;
using namespace nanos;
using namespace nanos::ext;

#define BARR_NUM 10

/*! works only with bf scheduler */
void barrier_code ( void * )
{
       for ( int i = 0; i < BARR_NUM; i++ ) {
              cout << "Before the " << i << " barrier" << endl;
              nanos_team_barrier();
              cout << "After the " << i << " barrier" << endl;

       }
}

int main (int argc, char **argv)
{
       cout << "start" << endl;
       //all threads perform a barrier: 
       ThreadTeam &team = *myThread->getTeam();
       for ( unsigned i = 1; i < team.size(); i++ ) {
              WD * wd = new WD(new SMPDD(barrier_code));
	      wd->tieTo(team[i]);
              sys.submit(*wd);
       }
       usleep(100);

       WD *wd = myThread->getCurrentWD();
       wd->tieTo(*myThread);
       barrier_code(NULL);

       cout << "end" << endl;
}
