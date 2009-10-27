#include "config.hpp"
#include "nanos.h"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"
#include <string.h>

using namespace std;
using namespace nanos;

void single_code (void *a)
{
	bool b;
	for (int i=0; i<1000; i++){
		nanos_single_guard(b);
		if (b)
		{
			cerr << "it: " << i << " th: " << myThread->getId() << endl ;
			usleep(10);
		}
	}
}

int main (int argc, char **argv)
{
	cout << "start" << endl;
	
	for ( int i = 1; i < sys.getNumPEs(); i++ ) {
		WD * wd = new WD(new SMPDD(single_code));
		sys.submit(*wd);
	}
	usleep(100);
	single_code(0);
	
	cout << "end" << endl;
}
