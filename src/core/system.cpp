#include "system.hpp"
#include "coresetup.hpp"
#include "smpprocessor.hpp"
#include "plugin.hpp"

using namespace nanos;

System nanos::sys;

System::System ()
{
   verbose0 ( "NANOS++ initalizing... start" );
   verbose0 ( "Preparing configuration" );
   CoreSetup::prepareConfig ( config );
   SMPProcessor::prepareConfig ( config );
   verbose0 ( "Reading Configuration" );
   config.init();
   verbose0 ( "Loading modules" );
   loadModules();
   verbose0 ( "Starting threads" );
   start();
   verbose0 ( "NANOS++ initalizing... end" );
}

void System::loadModules ()
{
// load default schedule plugin
   verbose0("loading " << CoreSetup::getDefaultSchedule());
   if ( !PluginManager::load ( "sched-"+CoreSetup::getDefaultSchedule() ) )
      fatal0 ( "Couldn't load main scheduling policy" );

   
}

SchedulingGroup * createBreadthFirstPolicy();
SchedulingGroup * createTaskStealPolicy ( int );
SchedulingGroup * createWFPolicy ( int, int, int, bool );
#define LIFO 1
#define FIFO 0

void System::start ()
{
   int numPes = CoreSetup::getNumPEs();

   // if preload, TODO: allow dynamic PE creation

   pes.reserve ( numPes );

   //TODO: remove, initialize policy dynamically
   SchedulingGroup *sg = createBreadthFirstPolicy();

   //TODO: decide, single master, multiple master start
   PE *pe = new SMPProcessor ( 0 );
   pes.push_back ( pe );
   pe->associateThisThread ( sg );

   for ( int p = 1; p < numPes ; p++ ) {
      // TODO: create processor type based on config
      pe = new SMPProcessor ( p );
      pes.push_back ( pe );

      //starting as much threads per pe as requested by the user

      for ( int ths = 0; ths < CoreSetup::getThsPerPE(); ths++ ) {
         pe->startWorker ( sg );
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
