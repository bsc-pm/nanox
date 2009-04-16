#include "system.hpp"
#include "coresetup.hpp"
#include "smpprocessor.hpp"
#include "schedule.hpp"

using namespace nanos;

System nanos::sys;

System::System ()
{
  CoreSetup::prepareConfig(config);

  init();
  start();
}

System::~System ()
{
}

void System::init ()
{
    config.init();
}

SchedulingGroup * createBreadthFirstPolicy();

void System::start ()
{
    int numPes = CoreSetup::getNumPEs();

    // if preload, TODO: allow dynamic PE creation
    pes.reserve(numPes);

    //TODO: remove, initialize policy dynamically
    SchedulingGroup *sg =  createBreadthFirstPolicy();
    //TODO: decide, single master, multiple master start

    // TODO: create self-worker
    pes[0] = new SMPProcessor(0,sg);
    pes[0]->associateThisThread();
    for ( int p = 1; p < numPes ; p++ ) {
	// TODO: create processor type based on config
	pes[p] = new SMPProcessor(p,sg);
	pes[p]->startWorker();
    }
}

//TODO: remove?
void System::submit (WD &work)
{
    Scheduler::submit(work);
}

//TODO: void system_description ()
// {
// 	// Altix simple
// 	System = new UMASystem ();
// 	System.addN(128, SmpPE(1594));
//
// 	// Altix complex
// 	System = new NUMASystem();
// 	System.add(64, Node(MultiCore(2,SmpPE(1594)),1GB) );
//
// 	// Cell
// 	System = new NUMASystem();
// 	System.add(1,Node(MultiCore(2,SmpPE(1000)));
// 	System.add(8,Node(SpuPE(),256K));
//
// 	// MN Partition
// 	System = new ClusterSystem();
// 	System.add(10,UMASystem(2,Multicore(2,SmpPE(1000))));
//
// 	// MI Partition
// 	System = new ClusterSystem();
// 	System.add(10,NUMASystem(1,MultiCore(2,SmpPE(2000)),16,SpuPE());
// }
