#include "coresetup.hpp"

using namespace nanos;

// Initialize this variables in prepareConfig to avoid problems between the order of constructors
int   CoreSetup::numPEs;
bool  CoreSetup::binding;
bool  CoreSetup::profile;
bool  CoreSetup::instrument;
bool  CoreSetup::verbose;
int   CoreSetup::thsPerPE;
std::string CoreSetup::defSchedule;
CoreSetup::ExecutionMode CoreSetup::executionMode;

void CoreSetup::prepareConfig ( Config &config )
{
   numPEs = 1;
   config.registerArgOption ( new Config::PositiveVar ( "nth-pes", numPEs ) );
   config.registerEnvOption ( new Config::PositiveVar ( "NTH_PES", numPEs ) );

   binding = false;
   config.registerArgOption ( new Config::FlagOption ( "nth-bindig", binding ) );
   
   verbose = false;
   config.registerArgOption ( new Config::FlagOption ( "nth-verbose", verbose ) );

   //more than 1 thread per pe
   thsPerPE=1;
   config.registerArgOption ( new Config::PositiveVar ( "nth-thsperpe", thsPerPE ) );

   new(&defSchedule) std::string("cilk");
   config.registerArgOption ( new Config::StringVar ( "nth-schedule", defSchedule ) );
   config.registerEnvOption ( new Config::StringVar ( "NTH_SCHEDULE", defSchedule ) );

   //TODO: how to simplify this a bit?
   executionMode = DEDICATED;
   Config::MapVar<ExecutionMode>::MapList opts ( 2 );
   opts[0] = Config::MapVar<ExecutionMode>::MapOption ( "dedicated", DEDICATED );
   opts[1] = Config::MapVar<ExecutionMode>::MapOption ( "shared", SHARED );
   config.registerArgOption (
      new Config::MapVar<ExecutionMode> ( "nth-mode", executionMode, opts ) );

}

