#include "system.hpp"
#include "config.hpp"
#include "plugin.hpp"
#include "schedule.hpp"
#include "smpprocessor.hpp"

using namespace nanos;

System nanos::sys;

// default system values go here
 System::System () : numPEs(1), binding(true), profile(false), instrument(false),
                     verboseMode(false), executionMode(DEDICATED), thsPerPE(1),
                     defSchedule("cilk")
{
   verbose0 ( "NANOS++ initalizing... start" );
   config();   
   loadModules();
   start();
   verbose0 ( "NANOS++ initalizing... end" );
}

void System::loadModules ()
{
   verbose0 ( "Loading modules" );
// load default schedule plugin
   verbose0("loading " << getDefaultSchedule());
   if ( !PluginManager::load ( "sched-"+getDefaultSchedule() ) )
      fatal0 ( "Couldn't load main scheduling policy" );

   ensure(defSGFactory,"No default system factory");
}

void System::config ()
{
   Config config;

   verbose0 ( "Preparing configuration" );
   config.registerArgOption(new Config::PositiveVar("nth-pes",numPEs));
   config.registerEnvOption(new Config::PositiveVar("NTH_PES",numPEs));
   config.registerArgOption(new Config::FlagOption("nth-bindig",binding));
   config.registerArgOption(new Config::FlagOption("nth-verbose",verboseMode));

   //more than 1 thread per pe
   config.registerArgOption(new Config::PositiveVar("nth-thsperpe",thsPerPE));

    //TODO: how to simplify this a bit?
   Config::MapVar<ExecutionMode>::MapList opts(2);
   opts[0] = Config::MapVar<ExecutionMode>::MapOption("dedicated",DEDICATED);
   opts[1] = Config::MapVar<ExecutionMode>::MapOption("shared",SHARED);
   config.registerArgOption(
                            new Config::MapVar<ExecutionMode>("nth-mode",executionMode,opts));

   config.registerArgOption ( new Config::StringVar ( "nth-schedule", defSchedule ) );
   config.registerEnvOption ( new Config::StringVar ( "NTH_SCHEDULE", defSchedule ) );
   
   SMPProcessor::prepareConfig(config);
   verbose0 ( "Reading Configuration" );
   config.init();
}

void System::start ()
{
   verbose0 ( "Starting threads" );

    int numPes = getNumPEs();

   // if preload, TODO: allow dynamic PE creation

   pes.reserve ( numPes );

   SchedulingGroup *sg = defSGFactory();

   //TODO: decide, single master, multiple master start
   PE *pe = new SMPProcessor ( 0 );
   pes.push_back ( pe );
   pe->associateThisThread ( sg );

    //starting as much threads per pe as requested by the user
    for(int ths = 1; ths < getThsPerPE(); ths++) {
         pe->startWorker(sg);
     }

    for ( int p = 1; p < numPes ; p++ ) {
      // TODO: create processor type based on config
      pe = new SMPProcessor ( p );
      pes.push_back ( pe );

      //starting as much threads per pe as requested by the user
      for(int ths = 0; ths < getThsPerPE(); ths++) {
         pe->startWorker(sg);
      }
   }
}

System::~System ()
{
   verbose ( "NANOS++ shutting down.... init" );

   verbose ( "Wait for main workgroup to complete" );
   myThread->getCurrentWD()->waitCompletation();

   verbose ( "Joining threads... phase 1" );
   // signal stop PEs

   for ( unsigned p = 1; p < pes.size() ; p++ ) {
      pes[p]->stopAll();
   }

   verbose ( "Joining threads... phase 2" );

   // join

   for ( unsigned p = 1; p < pes.size() ; p++ ) {
      delete pes[p];
   }

   verbose ( "NANOS++ shutting down.... end" );
}

//TODO: remove?
void System::submit ( WD &work )
{
   work.setParent ( myThread->getCurrentWD() );
   Scheduler::submit ( work );
}

//TODO: void system_description ()
// {
//  // Altix simple
//  System = new UMASystem ();
//  System.addN(128, SmpPE(1594));
//
//  // Altix complex
//  System = new NUMASystem();
//  System.add(64, Node(MultiCore(2,SmpPE(1594)),1GB) );
//
//  // Cell
//  System = new NUMASystem();
//  System.add(1,Node(MultiCore(2,SmpPE(1000)));
//  System.add(8,Node(SpuPE(),256K));
//
//  // MN Partition
//  System = new ClusterSystem();
//  System.add(10,UMASystem(2,Multicore(2,SmpPE(1000))));
//
//  // MI Partition
//  System = new ClusterSystem();
//  System.add(10,NUMASystem(1,MultiCore(2,SmpPE(2000)),16,SpuPE());
// }
