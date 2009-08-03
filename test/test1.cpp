#include "config.hpp"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"
#include <string.h>

using namespace std;
using namespace nanos;

int a = 1234;
std::string b("default");
bool c = false;

#if 0
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
#endif

typedef struct {
  int a;
  std::string b;
} hello_world_args; 

void hello_world (void *args)
{
  hello_world_args *hargs = (hello_world_args *) args;
  cout << "hello_world "
      << hargs->a << " "
      << hargs->b
      << endl;
}

int main (int argc, char **argv)
{
	cout << "PEs = " << sys.getNumPEs() << endl;
	cout << "Mode = " << sys.getExecutionMode() << endl;
	cout << "Verbose = " << sys.getVerbose() << endl;

	cout << "Args" << endl;
	for (int i = 0; i < argc; i++)
		cout << argv[i] << endl;

	cout << "start" << endl;

	const char *a = "alex";
	hello_world_args *data = new hello_world_args();
	data->a = 1;
	data->b = a;
	SMPWD * wd = new SMPWD(hello_world,data);
	a = "pepe";
	data = new hello_world_args();
	data->a = 2;
	data->b = a;
	SMPWD * wd2 = new SMPWD(hello_world,data);

	WG *wg = myThread->getCurrentWD();
	wg->addWork(*wd);
	wg->addWork(*wd2);
	sys.submit(*wd);
	sys.submit(*wd2);
	usleep(500);
	
	cout << "end" << endl;
}
