#include "config.hpp"
#include <iostream>
#include "coresetup.hpp"
#include "smpprocessor.hpp"
#include "system.hpp"
#include <string.h>

using namespace std;
using namespace nanos;

int a = 1234;
std::string b("default");
bool c = false;

class OMPModule {

class SetThreadsOption : public Config::PositiveAction
{
public:
	SetThreadsOption(const char *name) : Config::PositiveAction(name) {}
 	virtual void setValue (const int &value) { CoreSetup::setNumPEs(value); }
};

public:

void prepareConfig ()
{
 	//config.registerEnvOption(new SetThreadsOption("OMP_NUM_THREADS"));

	//config.registerEnvOption(new MapOption<std::string>("OMP_SCHEDULE",));
// 	if (getUnsignedEnvVar("OMP_STACK_SIZE",n)) {
// 		// TODO
// 	}*/
// 	
// 	//"OMP_SCHEDULE"
}

};

void hello_world (WD *wd)
{
  cout << "hello_world "
      << wd->getValue<int>(0) << " "
      << wd->getReference<char>(1)
      << endl;
}


int main (int argc, char **argv)
{


	cout << "PEs = " << CoreSetup::getNumPEs() << endl;
	cout << "Mode = " << CoreSetup::getExecutionMode() << endl;
	cout << "Verbose = " << CoreSetup::getVerbose() << endl;

//  	for ( int i = 0; i < CoreSetup::getNumPEs(); i++ ) {
//  		PE &pe = System.selectPE(i);
//  
//  		pe.startWorkingThread();
//  	}

	cout << "Args" << endl;
	for (int i = 0; i < argc; i++)
		cout << argv[i] << endl;

	cout << "start" << endl;

	const char *a = "alex";
	WorkData *data = new WorkData();
	data->setArguments(20,1,1,1,5,a);
        SMPWD * wd = new SMPWD(hello_world,data);
	a = "pepe";
	data = new WorkData();
	data->setArguments(20,1,1,2,5,a);
	SMPWD * wd2 = new SMPWD(hello_world,data);

//	WG *wg = myPE->getCurrentWD();
//	wd2->addWork(*wd);
//	wg->addWork(*wd2);
	sys.submit(*wd);
	sys.submit(*wd2);
	usleep(500);
	
	cout << "end" << endl;
}
