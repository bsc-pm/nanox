#include "coresetup.hpp"

using namespace nanos;

int 	CoreSetup::numPEs = 1;
bool 	CoreSetup::binding = false;
bool	CoreSetup::profile = true;
bool 	CoreSetup::instrument = false;
bool 	CoreSetup::verbose = false;

//more than 1 thread per pe
int CoreSetup::thsPerPE = 1;

CoreSetup::ExecutionMode CoreSetup::executionMode = DEDICATED;

void CoreSetup::prepareConfig (Config &config)
{
	config.registerArgOption(new Config::PositiveVar("nth-pes",numPEs));
	config.registerArgOption(new Config::FlagOption("nth-bindig",binding));
	config.registerArgOption(new Config::FlagOption("nth-verbose",verbose));

   //more than 1 thread per pe
   config.registerArgOption(new Config::PositiveVar("nth-thsperpe",thsPerPE));


	//TODO: how to simplify this a bit?
	Config::MapVar<ExecutionMode>::MapList opts(2);
	opts[0] = Config::MapVar<ExecutionMode>::MapOption("dedicated",DEDICATED);
	opts[1] = Config::MapVar<ExecutionMode>::MapOption("shared",SHARED);
	config.registerArgOption(
		new Config::MapVar<ExecutionMode>("nth-mode",executionMode,opts));

	// TODO: schedule
// 	vector<string> opts(3);
// 	opts[0] = "a";
// 	opts[1] = "b";
// 	opts[2] = "c";
// 
// 	config.registerEnvOption(
// 		new Config::MapVar<string>("NTH_SCHEDULE", schedule, opts ));
}

