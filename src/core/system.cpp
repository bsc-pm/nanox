#include "system.hpp"
#include "config.hpp"
#include "plugin.hpp"
#include "schedule.hpp"
#include "barrier.hpp"

using namespace nanos;

System nanos::sys;

//cutoff * createDummyCutoff();
//class dummy_cutoff;
//cutoff * createLevelCutoff();
//class level_cutoff;
//cutoff * createTasknumCutoff();
//class tasknum_cutoff;
//cutoff * createIdleCutoff();
//class idle_cutoff;
//cutoff * createReadyCutoff();
//class ready_cutoff;

class centralizedBarrier;
Barrier * createCentralizedBarrier(int);


// default system values go here
 System::System () : numPEs(1), binding(true), profile(false), instrument(false),
                     verboseMode(false), executionMode(DEDICATED), thsPerPE(1),
                     defSchedule("cilk"), defCutoff("tasknum")
{
    //cutOffPolicy = createDummyCutoff();
    verbose0 ( "NANOS++ initalizing... start" );
    config();   
    loadModules();
    start();
    verbose0 ( "NANOS++ initalizing... end" );
}

void System::loadModules ()
{
   verbose0 ( "Loading modules" );

   // load host processor module
   verbose0("loading SMP support");
   if ( !PluginManager::load ( "pe-smp" ) )
      fatal0 ( "Couldn't load SMP support" );
   ensure(hostFactory,"No default smp factory");

   // load default schedule plugin
   verbose0("loading " << getDefaultSchedule() << " scheduling policy support");
   if ( !PluginManager::load ( "sched-"+getDefaultSchedule() ) )
      fatal0 ( "Couldn't load main scheduling policy" );

   ensure(defSGFactory,"No default system scheduling factory");

   verbose0( "loading task num cutoff policy" );
   if( !PluginManager::load( "cutoff-"+getDefaultCutoff() ) )
      fatal0( "Could not load main cutoff policy" );
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

   config.registerArgOption ( new Config::StringVar ( "nth-cutoff", defCutoff ) );
   config.registerEnvOption ( new Config::StringVar ( "NTH_CUTOFF", defCutoff ) );

   verbose0 ( "Reading Configuration" );
   config.init();
}

PE * System::createPE ( std::string pe_type, int pid )
{
   // TODO: lookup table for PE factories
   // in the mean time assume only one factory

   return hostFactory(pid);
}

void System::start ()
{
   verbose0 ( "Starting threads" );

    int numPes = getNumPEs();

   pes.reserve ( numPes );

   SchedulingGroup *sg = defSGFactory(numPes*getThsPerPE());


   //currently, embedded barrier
   //TODO: move it to pluging
   sg->setBarrierImpl(createCentralizedBarrier(numPes*getThsPerPE()));


   //TODO: decide, single master, multiple master start
   PE *pe = createPE ( "smp", 0 );
   pes.push_back ( pe );
   workers.push_back(&pe->associateThisThread ( sg ));

   
    //starting as much threads per pe as requested by the user
    for(int ths = 1; ths < getThsPerPE(); ths++) {
         pe->startWorker(sg);
     }

    for ( int p = 1; p < numPes ; p++ ) {
      pe = createPE ( "smp", p );
      pes.push_back ( pe );

      //starting as much threads per pe as requested by the user
      for(int ths = 0; ths < getThsPerPE(); ths++) {
         workers.push_back(&pe->startWorker(sg));
      }
   }

   createTeam(numPes*getThsPerPE(),*sg,NULL,true);
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

void System::submit ( WD &work )
{
   work.setParent ( myThread->getCurrentWD() );
   work.setLevel( work.getParent()->getLevel() +1 );
   Scheduler::submit ( work );
}

void System::inlineWork ( WD &work )
{
   BaseThread *myself = myThread;
   
  // TODO: choose actual device...
  work.setParent ( myself->getCurrentWD() );
  myself->inlineWork(&work);
}


bool System::throttleTask() {
  return cutOffPolicy->cutoff_pred();
}


BaseThread * System:: getUnassignedWorker ( void )
{
    BaseThread *thread;
    
    for ( unsigned i  = 0; i < workers.size(); i++ ) {
       if ( !workers[i]->hasTeam() ) {
            thread = workers[i];
            // recheck availability with exclusive access
            thread->lock();
            if ( thread->hasTeam() ) {
               // we lost it
               thread->unlock();
               continue;
            }
            thread->reserve(); // set team flag only
            thread->unlock();

            return thread;
       }
    }

    return NULL;
}

void System::releaseWorker ( BaseThread * thread )
{
    //TODO: destroy if too many?
    thread->leaveTeam();
}

ThreadTeam * System:: createTeam (int nthreads, SG &policy, void *constraints, bool reuseCurrent)
{
     // create team
     ThreadTeam * team = new ThreadTeam(nthreads,policy);

     debug("Creating team " << team << " of " << nthreads << " threads");
     // find threads
     if ( reuseCurrent ) {
        nthreads --;
        team->addThread(myThread);
        myThread->enterTeam(team);
     }
     
     while (nthreads > 0) {
        BaseThread *thread = getUnassignedWorker();
        if (!thread) {
           // TODO: create one?
           break;
        }
        debug("adding thread " << thread << " to " << team);
        team->addThread(thread);
        thread->enterTeam(team);
     }

     return team;
}

