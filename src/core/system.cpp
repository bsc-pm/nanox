/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include "system.hpp"
#include "config.hpp"
#include "plugin.hpp"
#include "schedule.hpp"
#include "barrier.hpp"
#include "nanos-int.h"
#include "copydata.hpp"
#include "os.hpp"
#include "basethread.hpp"
#include "malign.hpp"
#include "processingelement.hpp"
#include "allocator.hpp"
#include "debug.hpp"
#include "dlb.hpp"
#include <string.h>
#include <set>
#include <climits>

#ifdef SPU_DEV
#include "spuprocessor.hpp"
#endif

#ifdef GPU_DEV
#include "gpuprocessor_decl.hpp"
#endif

#ifdef OpenCL_DEV
#include "openclprocessor.hpp"
#endif

using namespace nanos;

System nanos::sys;

// default system values go here
System::System () :
      _atomicWDSeed( 1 ), _threadIdSeed( 0 ),
      _numPEs( INT_MAX ), _numThreads( 0 ), _deviceStackSize( 0 ), _bindingStart (0), _bindingStride(1),  _bindThreads( true ), _profile( false ),
      _instrument( false ), _verboseMode( false ), _summary( false ), _executionMode( DEDICATED ), _initialMode( POOL ),
      _untieMaster( true ), _delayedStart( false ), _useYield( true ), _synchronizedStart( true ),
      _numSockets( 0 ), _coresPerSocket( 0 ), _numAvailSockets( 0 ), _enable_dlb( false ), _throttlePolicy ( NULL ),
      _schedStats(), _schedConf(), _defSchedule( "bf" ), _defThrottlePolicy( "hysteresis" ), 
      _defBarr( "centralized" ), _defInstr ( "empty_trace" ), _defDepsManager( "plain" ), _defArch( "smp" ),
      _initializedThreads ( 0 ), _targetThreads ( 0 ), _pausedThreads( 0 ),
      _pausedThreadsCond(), _unpausedThreadsCond(),
      _instrumentation ( NULL ), _defSchedulePolicy( NULL ), _dependenciesManager( NULL ),
      _pmInterface( NULL ), _useCaches( true ), _cachePolicy( System::DEFAULT ), _cacheMap()

#ifdef GPU_DEV
      , _pinnedMemoryCUDA( NEW CUDAPinnedMemoryManager() )
#endif
#ifdef NANOS_INSTRUMENTATION_ENABLED
      , _enableEvents(), _disableEvents(), _instrumentDefault("default"), _enable_cpuid_event( false )
#endif
      , _lockPoolSize(37), _lockPool( NULL ), _mainTeam (NULL)
{
   verbose0 ( "NANOS++ initializing... start" );

   // OS::init must be called here and not in System::start() as it can be too late
   // to locate the program arguments at that point
   OS::init();
   config();

   OS::getProcessAffinity( &_cpu_set );

   int cpu_count = getCpuCount();

   std::vector<int> cpu_affinity;
   cpu_affinity.reserve( cpu_count );
   std::ostringstream oss_cpu_idx;
   oss_cpu_idx << "[";
   for ( int i=0; i<CPU_SETSIZE; i++ ) {
      if ( CPU_ISSET(i, &_cpu_set) ) {
         cpu_affinity.push_back(i);
         oss_cpu_idx << i << ", ";
      }
   }
   oss_cpu_idx << "]";
   
   verbose0("PID[" << getpid() << "]. CPU affinity " << oss_cpu_idx.str());
   
   // Ensure everything is properly configured
   if( getNumPEs() == INT_MAX && _numThreads == 0 )
      // If no parameter specified, use all available CPUs
      setNumPEs( cpu_count );
   
   if ( _numThreads == 0 )
      // No threads specified? Use as many as PEs
      _numThreads = _numPEs;
   else if ( getNumPEs() == INT_MAX ){
      // No number of PEs given? Use 1 thread per PE
      setNumPEs(  _numThreads );
   }

   // Set _bindings structure once we have the system mask and the binding info
   _bindings.reserve( cpu_count );
   for ( int i=0, collisions = 0; i < cpu_count; ) {

      // The cast over cpu_affinity is needed because std::vector::size() returns a size_t type
      int pos = (_bindingStart + _bindingStride*i + collisions) % (int)cpu_affinity.size();

      // 'pos' may be negative if either bindingStart or bindingStride were negative
      // this loop fixes that % operator is the remainder, not the modulo operation
      while ( pos < 0 ) pos+=cpu_affinity.size();

      if ( std::find( _bindings.begin(), _bindings.end(), cpu_affinity[pos] ) != _bindings.end() ) {
         collisions++;
         ensure( collisions != cpu_count, "Reached limit of collisions. We should never get here." );
         continue;
      }
      _bindings.push_back( cpu_affinity[pos] );
      i++;
   }

   CPU_ZERO( &_cpu_active_set );

   _lockPool = NEW Lock[_lockPoolSize];

   if ( !_delayedStart ) {
      start();
   }
   verbose0 ( "NANOS++ initializing... end" );
}

struct LoadModule
{
   void operator() ( const char *module )
   {
      if ( module ) {
        verbose0( "loading " << module << " module" );
        sys.loadPlugin(module);
      }
   }
};

void System::loadModules ()
{
   verbose0 ( "Configuring module manager" );

   _pluginManager.init();

   verbose0 ( "Loading modules" );

   const OS::ModuleList & modules = OS::getRequestedModules();
   std::for_each(modules.begin(),modules.end(), LoadModule());

   // load host processor module
   if ( _hostFactory == NULL ) {
     verbose0( "loading Host support" );

     if ( !loadPlugin( "pe-"+getDefaultArch() ) )
       fatal0 ( "Couldn't load host support" );
   }
   ensure0( _hostFactory,"No default host factory" );

#ifdef GPU_DEV
   verbose0( "loading GPU support" );

   if ( !loadPlugin( "pe-gpu" ) )
      fatal0 ( "Couldn't load GPU support" );
#endif
   
#ifdef OpenCL_DEV
   verbose0( "loading OpenCL support" );

   if ( !loadPlugin( "pe-opencl" ) )
     fatal0 ( "Couldn't load OpenCL support" );
#endif
   
   // load default schedule plugin
   verbose0( "loading " << getDefaultSchedule() << " scheduling policy support" );

   if ( !loadPlugin( "sched-"+getDefaultSchedule() ) )
      fatal0 ( "Couldn't load main scheduling policy" );

   ensure0( _defSchedulePolicy,"No default system scheduling factory" );

   verbose0( "loading " << getDefaultThrottlePolicy() << " throttle policy" );

   if ( !loadPlugin( "throttle-"+getDefaultThrottlePolicy() ) )
      fatal0( "Could not load main cutoff policy" );

   ensure0( _throttlePolicy, "No default throttle policy" );

   verbose0( "loading " << getDefaultBarrier() << " barrier algorithm" );

   if ( !loadPlugin( "barrier-"+getDefaultBarrier() ) )
      fatal0( "Could not load main barrier algorithm" );

   if ( !loadPlugin( "instrumentation-"+getDefaultInstrumentation() ) )
      fatal0( "Could not load " + getDefaultInstrumentation() + " instrumentation" );

   ensure0( _defBarrFactory,"No default system barrier factory" );
   
   // load default dependencies plugin
   verbose0( "loading " << getDefaultDependenciesManager() << " dependencies manager support" );

   if ( !loadPlugin( "deps-"+getDefaultDependenciesManager() ) )
      fatal0 ( "Couldn't load main dependencies manager" );

   ensure0( _dependenciesManager,"No default dependencies manager" );

}

void System::unloadModules ()
{   
   delete _throttlePolicy;
   
   delete _defSchedulePolicy;
   
   //! \todo (#613): delete GPU plugin?
}

// Config Functor
struct ExecInit
{
   std::set<void *> _initialized;

   ExecInit() : _initialized() {}

   void operator() ( const nanos_init_desc_t & init )
   {
      if ( _initialized.find( (void *)init.func ) == _initialized.end() ) {
         init.func( init.data );
         _initialized.insert( ( void * ) init.func );
      }
   }
};

void System::config ()
{
   Config cfg;

   const OS::InitList & externalInits = OS::getInitializationFunctions();
   std::for_each(externalInits.begin(),externalInits.end(), ExecInit());
   
   if ( !_pmInterface ) {
      // bare bone run
      _pmInterface = NEW PMInterface();
   }

   //! Declare all configuration core's flags
   verbose0( "Preparing library configuration" );

   cfg.setOptionsSection( "Core", "Core options of the core of Nanos++ runtime" );

   cfg.registerConfigOption( "num_pes", NEW Config::UintVar( _numPEs ),
                             "Defines the number of processing elements" );
   cfg.registerArgOption( "num_pes", "pes" );
   cfg.registerEnvOption( "num_pes", "NX_PES" );

   cfg.registerConfigOption( "num_threads", NEW Config::PositiveVar( _numThreads ),
                             "Defines the number of threads. Note that OMP_NUM_THREADS is an alias to this." );
   cfg.registerArgOption( "num_threads", "threads" );
   cfg.registerEnvOption( "num_threads", "NX_THREADS" );
   
   cfg.registerConfigOption( "cores-per-socket", NEW Config::PositiveVar( _coresPerSocket ),
                             "Number of cores per socket." );
   cfg.registerArgOption( "cores-per-socket", "cores-per-socket" );
   
   cfg.registerConfigOption( "num-sockets", NEW Config::PositiveVar( _numSockets ),
                             "Number of sockets available." );
   cfg.registerArgOption( "num-sockets", "num-sockets" );
   
   cfg.registerConfigOption( "hwloc-topology", NEW Config::StringVar( _topologyPath ),
                             "Overrides hwloc's topology discovery and uses the one provided by an XML file." );
   cfg.registerArgOption( "hwloc-topology", "hwloc-topology" );
   cfg.registerEnvOption( "hwloc-topology", "NX_HWLOC_TOPOLOGY_PATH" );
   

   cfg.registerConfigOption( "stack-size", NEW Config::PositiveVar( _deviceStackSize ),
                             "Defines the default stack size for all devices" );
   cfg.registerArgOption( "stack-size", "stack-size" );
   cfg.registerEnvOption( "stack-size", "NX_STACK_SIZE" );

   cfg.registerConfigOption( "binding-start", NEW Config::IntegerVar ( _bindingStart ),
                             "Set initial cpu id for binding (binding required)" );
   cfg.registerArgOption( "binding-start", "binding-start" );
   cfg.registerEnvOption( "binding-start", "NX_BINDING_START" );

   cfg.registerConfigOption( "binding-stride", NEW Config::IntegerVar ( _bindingStride ),
                             "Set binding stride (binding required)" );
   cfg.registerArgOption( "binding-stride", "binding-stride" );
   cfg.registerEnvOption( "binding-stride", "NX_BINDING_STRIDE" );

   cfg.registerConfigOption( "no-binding", NEW Config::FlagOption( _bindThreads, false ),
                             "Disables thread binding" );
   cfg.registerArgOption( "no-binding", "disable-binding" );

   cfg.registerConfigOption( "no-yield", NEW Config::FlagOption( _useYield, false ),
                             "Do not yield on idle and condition waits" );
   cfg.registerArgOption( "no-yield", "disable-yield" );

   cfg.registerConfigOption( "verbose", NEW Config::FlagOption( _verboseMode ),
                             "Activates verbose mode" );
   cfg.registerArgOption( "verbose", "verbose" );

   cfg.registerConfigOption( "summary", NEW Config::FlagOption( _summary ),
                             "Activates summary mode" );
   cfg.registerArgOption( "summary", "summary" );

//! \bug implement execution modes (#146) */
#if 0
   cfg::MapVar<ExecutionMode> map( _executionMode );
   map.addOption( "dedicated", DEDICATED).addOption( "shared", SHARED );
   cfg.registerConfigOption ( "exec_mode", &map, "Execution mode" );
   cfg.registerArgOption ( "exec_mode", "mode" );
#endif

   registerPluginOption( "schedule", "sched", _defSchedule,
                         "Defines the scheduling policy", cfg );
   cfg.registerArgOption( "schedule", "schedule" );
   cfg.registerEnvOption( "schedule", "NX_SCHEDULE" );

   registerPluginOption( "throttle", "throttle", _defThrottlePolicy,
                         "Defines the throttle policy", cfg );
   cfg.registerArgOption( "throttle", "throttle" );
   cfg.registerEnvOption( "throttle", "NX_THROTTLE" );

   cfg.registerConfigOption( "barrier", NEW Config::StringVar ( _defBarr ),
                             "Defines barrier algorithm" );
   cfg.registerArgOption( "barrier", "barrier" );
   cfg.registerEnvOption( "barrier", "NX_BARRIER" );

   registerPluginOption( "instrumentation", "instrumentation", _defInstr,
                         "Defines instrumentation format", cfg );
   cfg.registerArgOption( "instrumentation", "instrumentation" );
   cfg.registerEnvOption( "instrumentation", "NX_INSTRUMENTATION" );

   cfg.registerConfigOption( "no-sync-start", NEW Config::FlagOption( _synchronizedStart, false),
                             "Disables synchronized start" );
   cfg.registerArgOption( "no-sync-start", "disable-synchronized-start" );

   cfg.registerConfigOption( "architecture", NEW Config::StringVar ( _defArch ),
                             "Defines the architecture to use (smp by default)" );
   cfg.registerArgOption( "architecture", "architecture" );
   cfg.registerEnvOption( "architecture", "NX_ARCHITECTURE" );

   cfg.registerConfigOption( "no-caches", NEW Config::FlagOption( _useCaches, false ), "Disables the use of caches" );
   cfg.registerArgOption( "no-caches", "disable-caches" );

   CachePolicyConfig *cachePolicyCfg = NEW CachePolicyConfig ( _cachePolicy );
   cachePolicyCfg->addOption("wt", System::WRITE_THROUGH );
   cachePolicyCfg->addOption("wb", System::WRITE_BACK );
   cachePolicyCfg->addOption( "nocache", System::NONE );

   cfg.registerConfigOption( "cache-policy", cachePolicyCfg,
                             "Defines the general cache policy to use: write-through / write-back. Can be overwritten for specific architectures" );
   cfg.registerArgOption( "cache-policy", "cache-policy" );
   cfg.registerEnvOption( "cache-policy", "NX_CACHE_POLICY" );
   
   registerPluginOption( "deps", "deps", _defDepsManager,
                         "Defines the dependencies plugin", cfg );
   cfg.registerArgOption( "deps", "deps" );
   cfg.registerEnvOption( "deps", "NX_DEPS" );
   

#ifdef NANOS_INSTRUMENTATION_ENABLED
   cfg.registerConfigOption( "instrument-default", NEW Config::StringVar ( _instrumentDefault ),
                             "Set instrumentation event list default (none, all)" );
   cfg.registerArgOption( "instrument-default", "instrument-default" );

   cfg.registerConfigOption( "instrument-enable", NEW Config::StringVarList ( _enableEvents ),
                             "Add events to instrumentation event list" );
   cfg.registerArgOption( "instrument-enable", "instrument-enable" );

   cfg.registerConfigOption( "instrument-disable", NEW Config::StringVarList ( _disableEvents ),
                             "Remove events to instrumentation event list" );
   cfg.registerArgOption( "instrument-disable", "instrument-disable" );

   cfg.registerConfigOption( "instrument-cpuid", NEW Config::FlagOption ( _enable_cpuid_event ),
                             "Add cpuid event when binding is disabled (expensive)" );
   cfg.registerArgOption( "instrument-cpuid", "instrument-cpuid" );
#endif

   cfg.registerConfigOption( "enable-dlb", NEW Config::FlagOption ( _enable_dlb ),
                              "Tune Nanos Runtime to be used with Dynamic Load Balancing library)" );
   cfg.registerArgOption( "enable-dlb", "enable-dlb" );

   _schedConf.config( cfg );
   _pmInterface->config( cfg );

   verbose0 ( "Reading Configuration" );

   cfg.init();
}

PE * System::createPE ( std::string pe_type, int pid, int uid )
{
   //! \todo lookup table for PE factories, in the mean time assume only one factory
   return _hostFactory( pid, uid );
}

void System::start ()
{
   if ( !_useCaches ) _cachePolicy = System::NONE;
   
   //! Load hwloc first, in order to make it available for modules
   if ( isHwlocAvailable() )
      loadHwloc();

   // loadNUMAInfo needs _targetThreads when hwloc is not available.
   // Note that it is not its final value!
   _targetThreads = _numThreads;
   
   // Load & check NUMA config
   loadNUMAInfo();

   // Modules can be loaded now
   loadModules();

   // Increase targetThreads, ask the architecture plugins
   for ( ArchitecturePlugins::const_iterator it = _archs.begin();
        it != _archs.end(); ++it )
   {
      _targetThreads += (*it)->getNumThreads();
   }

   // Instrumentation startup
   NANOS_INSTRUMENT ( sys.getInstrumentation()->filterEvents( _instrumentDefault, _enableEvents, _disableEvents ) );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->initialize() );

   verbose0 ( "Starting runtime" );

   _pmInterface->start();

   int numPes = getNumPEs();

   _pes.reserve ( numPes );

   PE *pe = createPE ( _defArch, getBindingId( 0 ), 0 );
   pe->setNUMANode( getNodeOfPE( pe->getId() ) );
   _pes.push_back ( pe );
   _workers.push_back( &pe->associateThisThread ( getUntieMaster() ) );
   CPU_SET( getBindingId( 0 ), &_cpu_active_set );

   WD &mainWD = *myThread->getCurrentWD();
   (void) mainWD.getDirectory(true);
   
   if ( _pmInterface->getInternalDataSize() > 0 ) {
      char *data = NEW char[_pmInterface->getInternalDataSize()];
      _pmInterface->initInternalData( data );
      mainWD.setInternalData( data );
   }

   _pmInterface->setupWD( mainWD );

   /* Renaming currend thread as Master */
   myThread->rename("Master");

   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent (NANOS_STARTUP) );
   
   // For each plugin, notify it's the way to reserve PEs if they are required
   for ( ArchitecturePlugins::const_iterator it = _archs.begin();
        it != _archs.end(); ++it )
   {
      (*it)->createBindingList();
   }   
   // Right now, _bindings should only store SMP PEs ids

   // Create PEs
   int p;
   for ( p = 1; p < numPes ; p++ ) {
      pe = createPE ( "smp", getBindingId( p ), p );
      pe->setNUMANode( getNodeOfPE( pe->getId() ) );
      _pes.push_back ( pe );

      CPU_SET( getBindingId( p ), &_cpu_active_set );
   }
   
   // Create threads
   for ( int ths = 1; ths < _numThreads; ths++ ) {
      pe = _pes[ ths % numPes ];
      _workers.push_back( &pe->startWorker() );
   }
   
   // For each plugin create PEs and workers
   //! \bug  FIXME (#855)
   for ( ArchitecturePlugins::const_iterator it = _archs.begin();
        it != _archs.end(); ++it )
   {
      for ( unsigned archPE = 0; archPE < (*it)->getNumPEs(); ++archPE )
      {
         PE * processor = (*it)->createPE( archPE, p );
         fatal_cond0( processor == NULL, "ArchPlugin::createPE returned NULL" );
         _pes.push_back( processor );
         _workers.push_back( &processor->startWorker() );
         CPU_SET( processor->getId(), &_cpu_active_set );
         ++p;
      }
   }

   // Set up internal data for each worker
   for ( ThreadList::const_iterator it = _workers.begin(); it != _workers.end(); it++ ) {

      WD & threadWD = (*it)->getThreadWD();
      if ( _pmInterface->getInternalDataSize() > 0 ) {
         char *data = NEW char[_pmInterface->getInternalDataSize()];
         _pmInterface->initInternalData( data );
         threadWD.setInternalData( data );
      }
      _pmInterface->setupWD( threadWD );
   }
      
#ifdef SPU_DEV
   PE *spu = NEW nanos::ext::SPUProcessor(100, (nanos::ext::SMPProcessor &) *_pes[0]);
   spu->startWorker();
#endif


   if ( getSynchronizedStart() ) {
      // Wait for the rest of the gang to be ready, but do not yet notify master thread is ready
      while (_initializedThreads.value() < ( _targetThreads - 1 ) ) {}
   }

   // FIXME (855): do this before thread creation, after PE creation
   completeNUMAInfo();

   switch ( getInitialMode() )
   {
      case POOL:
         verbose0("Pool model enabled (OmpSs)");
         _mainTeam = createTeam( _workers.size() );
         break;
      case ONE_THREAD:
         verbose0("One-thread model enabled (OpenMP)");
         _mainTeam = createTeam(1);
         break;
      default:
         fatal("Unknown initial mode!");
         break;
   }

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t num_threads_key = ID->getEventKey("set-num-threads"); )
   NANOS_INSTRUMENT ( nanos_event_value_t team_size =  (nanos_event_value_t) myThread->getTeam()->size(); )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &num_threads_key, &team_size); )
   
   /* Master thread is ready now */
   if ( getSynchronizedStart() )
     threadReady();

   // Paused threads: set the condition checker 
   _pausedThreadsCond.setConditionChecker( EqualConditionChecker<unsigned int >( &_pausedThreads.override(), _workers.size() ) );
   _unpausedThreadsCond.setConditionChecker( EqualConditionChecker<unsigned int >( &_pausedThreads.override(), 0 ) );

   // All initialization is ready, call postInit hooks
   const OS::InitList & externalInits = OS::getPostInitializationFunctions();
   std::for_each(externalInits.begin(),externalInits.end(), ExecInit());

   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent (NANOS_RUNNING) );
   
   // List unrecognised arguments
   std::string unrecog = Config::getOrphanOptions();
   if ( !unrecog.empty() )
      warning( "Unrecognised arguments: " << unrecog );
   Config::deleteOrphanOptions();
      
   // hwloc can be now unloaded
   if ( isHwlocAvailable() )
      unloadHwloc();

   if ( _summary )
      environmentSummary();
}

System::~System ()
{
   if ( !_delayedStart ) finish();
}

void System::finish ()
{
   /* Instrumentation: First removing RUNNING state from top of the state statck */
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent(NANOS_SHUTDOWN) );

   verbose ( "NANOS++ statistics");
   verbose ( std::dec << (unsigned int) getCreatedTasks() << " tasks has been executed" );

   verbose ( "NANOS++ shutting down.... init" );
   verbose ( "Wait for main workgroup to complete" );
   myThread->getCurrentWD()->waitCompletion( true );

   // we need to switch to the main thread here to finish
   // the execution correctly
   getMyThreadSafe()->getCurrentWD()->tied().tieTo(*_workers[0]);
   Scheduler::switchToThread(_workers[0]);
   
   ensure(getMyThreadSafe()->getId() == 0, "Main thread not finishing the application!");

   verbose ( "Joining threads... phase 1" );
   // signal stop PEs

   for ( unsigned p = 0; p < _pes.size() ; p++ ) {
      _pes[p]->stopAll();
   }

   verbose ( "Joining threads... phase 2" );

   // shutdown instrumentation
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->finalize() );

   ensure( _schedStats._readyTasks == 0, "Ready task counter has an invalid value!");

   _pmInterface->finish();
   delete _pmInterface;

   /* System mem free */

   delete[] _lockPool;

   /* deleting master WD */
   //getMyThreadSafe()->getCurrentWD()->~WorkDescriptor();
   delete (WorkDescriptor *) (getMyThreadSafe()->getCurrentWD());

   for ( Slicers::const_iterator it = _slicers.begin(); it !=   _slicers.end(); it++ ) {
      delete (Slicer *)  it->second;
   }

   for ( WorkSharings::const_iterator it = _worksharings.begin(); it !=   _worksharings.end(); it++ ) {
      delete (WorkSharing *)  it->second;
   }
   
   /* deleting thread team */
   ThreadTeam* team = getMyThreadSafe()->getTeam();

   if ( team->getScheduleData() != NULL ) team->getScheduleData()->printStats();

   /* For every thread in the team */
   while ( team->size() > 0 ) {
      BaseThread* pThread = team->popThread();
      pThread->leaveTeam();
   }
   delete team;

   // join
   for ( unsigned p = 1; p < _pes.size() ; p++ ) {
      delete _pes[p];
   }
   
   /* unload modules */
   unloadModules();

   delete _dependenciesManager;

   // Deleting last processing element
   delete _pes[0];

   if ( allocator != NULL ) free (allocator);

   verbose ( "NANOS++ shutting down.... end" );

   if ( _summary )
      executionSummary();
}

/*! \brief Creates a new WD
 *
 *  This function creates a new WD, allocating memory space for device ptrs and
 *  data when necessary. 
 *
 *  \param [in,out] uwd is the related addr for WD if this parameter is null the
 *                  system will allocate space in memory for the new WD
 *  \param [in] num_devices is the number of related devices
 *  \param [in] devices is a vector of device descriptors 
 *  \param [in] data_size is the size of the related data
 *  \param [in,out] data is the related data (allocated if needed)
 *  \param [in] uwg work group to relate with
 *  \param [in] props new WD properties
 *  \param [in] num_copies is the number of copy objects of the WD
 *  \param [in] copies is vector of copy objects of the WD
 *  \param [in] num_dimensions is the number of dimension objects associated to the copies
 *  \param [in] dimensions is vector of dimension objects
 *
 *  When it does a full allocation the layout is the following:
 *  <pre>
 *  +---------------+
 *  |     WD        |
 *  +---------------+
 *  |    data       |
 *  +---------------+
 *  |  dev_ptr[0]   |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |  dev_ptr[N]   |
 *  +---------------+
 *  |     DD0       |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |     DDN       |
 *  +---------------+
 *  |    copy0      |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |    copyM      |
 *  +---------------+
 *  |     dim0      |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |     dimM      |
 *  +---------------+
 *  |   PM Data     |
 *  +---------------+
 *  </pre>
 */
void System::createWD ( WD **uwd, size_t num_devices, nanos_device_t *devices, size_t data_size, size_t data_align,
                        void **data, WG *uwg, nanos_wd_props_t *props, nanos_wd_dyn_props_t *dyn_props,
                        size_t num_copies, nanos_copy_data_t **copies, size_t num_dimensions,
                        nanos_region_dimension_internal_t **dimensions, nanos_translate_args_t translate_args,
                        const char *description )
{
   ensure(num_devices > 0,"WorkDescriptor has no devices");

   unsigned int i;
   char *chunk = 0;

   size_t size_CopyData;
   size_t size_Data, offset_Data, size_DPtrs, offset_DPtrs, size_Copies, offset_Copies, size_Dimensions, offset_Dimensions, offset_PMD;
   size_t offset_DESC, size_DESC;
   char *desc;
   size_t total_size;

   // WD doesn't need to compute offset, it will always be the chunk allocated address

   // Computing Data info
   size_Data = (data != NULL && *data == NULL)? data_size:0;
   if ( *uwd == NULL ) offset_Data = NANOS_ALIGNED_MEMORY_OFFSET(0, sizeof(WD), data_align );
   else offset_Data = 0; // if there are no wd allocated, it will always be the chunk allocated address

   // Computing Data Device pointers and Data Devicesinfo
   size_DPtrs    = sizeof(DD *) * num_devices;
   offset_DPtrs  = NANOS_ALIGNED_MEMORY_OFFSET(offset_Data, size_Data, __alignof__( DD*) );

   // Computing Copies info
   if ( num_copies != 0 ) {
      size_CopyData = sizeof(CopyData);
      size_Copies   = size_CopyData * num_copies;
      offset_Copies = NANOS_ALIGNED_MEMORY_OFFSET(offset_DPtrs, size_DPtrs, __alignof__(nanos_copy_data_t) );
      // There must be at least 1 dimension entry
      size_Dimensions = num_dimensions * sizeof(nanos_region_dimension_internal_t);
      offset_Dimensions = NANOS_ALIGNED_MEMORY_OFFSET(offset_Copies, size_Copies, __alignof__(nanos_region_dimension_internal_t) );
   } else {
      size_Copies = 0;
      // No dimensions
      size_Dimensions = 0;
      offset_Copies = offset_Dimensions = NANOS_ALIGNED_MEMORY_OFFSET(offset_DPtrs, size_DPtrs, 1);
   }

   // Computing description char * + description
   if ( description == NULL ) {
      offset_DESC = offset_Dimensions;
      size_DESC = size_Dimensions;
   } else {
      offset_DESC = NANOS_ALIGNED_MEMORY_OFFSET(offset_Dimensions, size_Dimensions, __alignof__ (void*) );
      size_DESC = (strlen(description)+1) * sizeof(char);
   }

   // Computing Internal Data info and total size
   static size_t size_PMD   = _pmInterface->getInternalDataSize();
   if ( size_PMD != 0 ) {
      static size_t align_PMD = _pmInterface->getInternalDataAlignment();
      offset_PMD = NANOS_ALIGNED_MEMORY_OFFSET(offset_DESC, size_DESC, align_PMD);
      total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_PMD,size_PMD,1);
   } else {
      offset_PMD = 0; // needed for a gcc warning
      total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_DESC, size_DESC, 1);
   }

   chunk = NEW char[total_size];
   if (props->clear_chunk)
       memset(chunk, 0, sizeof(char) * total_size);

   // allocating WD and DATA
   if ( *uwd == NULL ) *uwd = (WD *) chunk;
   if ( data != NULL && *data == NULL ) *data = (chunk + offset_Data);

   // allocating Device Data
   DD **dev_ptrs = ( DD ** ) (chunk + offset_DPtrs);
   for ( i = 0 ; i < num_devices ; i ++ ) dev_ptrs[i] = ( DD* ) devices[i].factory( devices[i].arg );

   ensure ((num_copies==0 && copies==NULL && num_dimensions==0 && dimensions==NULL) || (num_copies!=0 && copies!=NULL && num_dimensions!=0 && dimensions!=NULL ), "Number of copies and copy data conflict" );
   

   // allocating copy-ins/copy-outs
   if ( copies != NULL && *copies == NULL ) {
      *copies = ( CopyData * ) (chunk + offset_Copies);
      *dimensions = ( nanos_region_dimension_internal_t * ) ( chunk + offset_Dimensions );
   }

   // Copying description string
   if ( description == NULL ) desc = NULL;
   else {
      desc = (chunk + offset_DESC);
      strncpy ( desc, description, size_DESC);
//      desc[strlen(description)]='\0';
   }

   WD * wd =  new (*uwd) WD( num_devices, dev_ptrs, data_size, data_align, data != NULL ? *data : NULL,
                             num_copies, (copies != NULL)? *copies : NULL, translate_args, desc );
   // Set WD's socket
   wd->setSocket( getCurrentSocket() );
   
   // Set total size
   wd->setTotalSize(total_size );
   
   if ( getCurrentSocket() >= sys.getNumSockets() )
      throw NANOS_INVALID_PARAM;

   // All the implementations for a given task will have the same ID
   wd->setVersionGroupId( ( unsigned long ) devices );

   // initializing internal data
   if ( size_PMD > 0) {
      _pmInterface->initInternalData( chunk + offset_PMD );
      wd->setInternalData( chunk + offset_PMD );
   }

   // add to workgroup
   if ( uwg != NULL ) {
      WG * wg = ( WG * )uwg;
      wg->addWork( *wd );
   }

   // set properties
   if ( props != NULL ) {
      if ( props->tied ) wd->tied();
      wd->setPriority( dyn_props->priority );
      wd->setFinal ( dyn_props->flags.is_final );
   }
   if ( dyn_props && dyn_props->tie_to ) wd->tieTo( *( BaseThread * )dyn_props->tie_to );
   
   /* DLB */
   // In case the master have been busy crating tasks 
   // every 10 tasks created I'll check available cpus
   if(_atomicWDSeed.value()%10==0)dlb_updateAvailableCpus();
}

/*! \brief Creates a new Sliced WD
 *
 *  \param [in,out] uwd is the related addr for WD if this parameter is null the
 *                  system will allocate space in memory for the new WD
 *  \param [in] num_devices is the number of related devices
 *  \param [in] devices is a vector of device descriptors 
 *  \param [in] outline_data_size is the size of the related data
 *  \param [in,out] outline_data is the related data (allocated if needed)
 *  \param [in] uwg work group to relate with
 *  \param [in] slicer is the related slicer which contains all the methods to manage this WD
 *  \param [in,out] data used as the slicer data (allocated if needed)
 *  \param [in] props new WD properties
 *
 *  \return void
 *
 *  \par Description:
 * 
 *  This function creates a new Sliced WD, allocating memory space for device ptrs and
 *  data when necessary. Also allocates Slicer Data object which is related with the WD.
 *
 *  When it does a full allocation the layout is the following:
 *  <pre>
 *  +---------------+
 *  |   slicedWD    |
 *  +---------------+
 *  |    data       |
 *  +---------------+
 *  |  dev_ptr[0]   |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |  dev_ptr[N]   |
 *  +---------------+
 *  |     DD0       |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |     DDN       |
 *  +---------------+
 *  |    copy0      |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |    copyM      |
 *  +---------------+
 *  |     dim0      |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |     dimM      |
 *  +---------------+
 *  |   PM Data     |
 *  +---------------+
 *  </pre>
 *
 * \sa createWD, duplicateWD, duplicateSlicedWD
 */
void System::createSlicedWD ( WD **uwd, size_t num_devices, nanos_device_t *devices, size_t outline_data_size,
                        int outline_data_align, void **outline_data, WG *uwg, Slicer *slicer, nanos_wd_props_t *props,
                        nanos_wd_dyn_props_t *dyn_props, size_t num_copies, nanos_copy_data_t **copies, size_t num_dimensions,
                        nanos_region_dimension_internal_t **dimensions, const char *description )
{
   ensure(num_devices > 0,"WorkDescriptor has no devices");

   unsigned int i;
   char *chunk = 0;

   size_t size_CopyData;
   size_t size_Data, offset_Data, size_DPtrs, offset_DPtrs;
   size_t size_Copies, offset_Copies, size_Dimensions, offset_Dimensions, offset_PMD;
   size_t offset_DESC, size_DESC;
   char *desc;
   size_t total_size;

   // WD doesn't need to compute offset, it will always be the chunk allocated address

   // Computing Data info
   size_Data = (outline_data != NULL && *outline_data == NULL)? outline_data_size:0;
   if ( *uwd == NULL ) offset_Data = NANOS_ALIGNED_MEMORY_OFFSET(0, sizeof(SlicedWD), outline_data_align );
   else offset_Data = 0; // if there are no wd allocated, it will always be the chunk allocated address

   // Computing Data Device pointers and Data Devicesinfo
   size_DPtrs    = sizeof(DD *) * num_devices;
   offset_DPtrs  = NANOS_ALIGNED_MEMORY_OFFSET(offset_Data, size_Data, __alignof__( DD*) );

   // Computing Copies info
   if ( num_copies != 0 ) {
      size_CopyData = sizeof(CopyData);
      size_Copies   = size_CopyData * num_copies;
      offset_Copies = NANOS_ALIGNED_MEMORY_OFFSET(offset_DPtrs, size_DPtrs, __alignof__(nanos_copy_data_t) );
      // There must be at least 1 dimension entry
      size_Dimensions = num_dimensions * sizeof(nanos_region_dimension_internal_t);
      offset_Dimensions = NANOS_ALIGNED_MEMORY_OFFSET(offset_Copies, size_Copies, __alignof__(nanos_region_dimension_internal_t) );
   } else {
      size_Copies = 0;
      // No dimensions
      size_Dimensions = 0;
      offset_Copies = offset_Dimensions = NANOS_ALIGNED_MEMORY_OFFSET(offset_DPtrs, size_DPtrs, 1);
   }

   // Computing description char * + description
   if ( description == NULL ) {
      offset_DESC = offset_Dimensions;
      size_DESC = size_Dimensions;
   } else {
      offset_DESC = NANOS_ALIGNED_MEMORY_OFFSET(offset_Dimensions, size_Dimensions, __alignof__ (void*) );
      size_DESC = (strlen(description)+1) * sizeof(char);
   }

   // Computing Internal Data info and total size
   static size_t size_PMD   = _pmInterface->getInternalDataSize();
   if ( size_PMD != 0 ) {
      static size_t align_PMD = _pmInterface->getInternalDataAlignment();
      offset_PMD = NANOS_ALIGNED_MEMORY_OFFSET(offset_DESC, size_DESC, align_PMD);
   } else {
      offset_PMD = NANOS_ALIGNED_MEMORY_OFFSET(offset_DESC, size_DESC, 1);
   }

   total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_PMD, size_PMD, 1);

   chunk = NEW char[total_size];

   // allocating WD and DATA
   if ( *uwd == NULL ) *uwd = (SlicedWD *) chunk;
   if ( outline_data != NULL && *outline_data == NULL ) *outline_data = (chunk + offset_Data);

   // allocating and initializing Device Data pointers
   DD **dev_ptrs = ( DD ** ) (chunk + offset_DPtrs);
   for ( i = 0 ; i < num_devices ; i ++ ) dev_ptrs[i] = ( DD* ) devices[i].factory( devices[i].arg );

   ensure ((num_copies==0 && copies==NULL && num_dimensions==0 && dimensions==NULL) || (num_copies!=0 && copies!=NULL && num_dimensions!=0 && dimensions!=NULL ), "Number of copies and copy data conflict" );

   // allocating copy-ins/copy-outs
   if ( copies != NULL && *copies == NULL ) {
      *copies = ( CopyData * ) (chunk + offset_Copies);
      *dimensions = ( nanos_region_dimension_internal_t * ) ( chunk + offset_Dimensions );
   }

   // Copying description string
   if ( description == NULL ) desc = NULL;
   else desc = (chunk + offset_DESC);

   SlicedWD * wd =  new (*uwd) SlicedWD( *slicer, num_devices, dev_ptrs, outline_data_size, outline_data_align,
                                         outline_data != NULL ? *outline_data : NULL, num_copies, (copies == NULL) ? NULL : *copies, desc );
   // Set WD's socket
   wd->setSocket(  getCurrentSocket() );

   // Set total size
   wd->setTotalSize(total_size );
   
   if ( getCurrentSocket() >= sys.getNumSockets() )
      throw NANOS_INVALID_PARAM;

   // initializing internal data
   if ( size_PMD > 0) {
      _pmInterface->initInternalData( chunk + offset_PMD );
      wd->setInternalData( chunk + offset_PMD );
   }

   // add to workgroup
   if ( uwg != NULL ) {
      WG * wg = ( WG * )uwg;
      wg->addWork( *wd );
   }

   // set properties
   if ( props != NULL ) {
      if ( props->tied ) wd->tied();
      wd->setPriority( dyn_props->priority );
      wd->setFinal ( dyn_props->flags.is_final );
   }
   if ( dyn_props && dyn_props->tie_to ) wd->tieTo( *( BaseThread * )dyn_props->tie_to );
}

/*! \brief Duplicates the whole structure for a given WD
 *
 *  \param [out] uwd is the target addr for the new WD
 *  \param [in] wd is the former WD
 *
 *  \return void
 *
 *  \par Description:
 *
 *  This function duplicates the given WD passed as a parameter copying all the
 *  related data included in the layout (devices ptr, data and DD). First it computes
 *  the size for the layout, then it duplicates each one of the chunks (Data,
 *  Device's pointers, internal data, etc). Finally calls WorkDescriptor constructor
 *  using new and placement.
 *
 *  \sa WorkDescriptor, createWD, createSlicedWD, duplicateSlicedWD
 */
void System::duplicateWD ( WD **uwd, WD *wd)
{
   unsigned int i, num_Devices, num_Copies, num_Dimensions;
   DeviceData **dev_data;
   void *data = NULL;
   char *chunk = 0, *chunk_iter;

   size_t size_CopyData;
   size_t size_Data, offset_Data, size_DPtrs, offset_DPtrs, size_Copies, offset_Copies, size_Dimensions, offset_Dimensions, offset_PMD;
   size_t total_size;

   // WD doesn't need to compute offset, it will always be the chunk allocated address

   // Computing Data info
   size_Data = wd->getDataSize();
   if ( *uwd == NULL ) offset_Data = NANOS_ALIGNED_MEMORY_OFFSET(0, sizeof(WD), wd->getDataAlignment() );
   else offset_Data = 0; // if there are no wd allocated, it will always be the chunk allocated address

   // Computing Data Device pointers and Data Devicesinfo
   num_Devices = wd->getNumDevices();
   dev_data = wd->getDevices();
   size_DPtrs    = sizeof(DD *) * num_Devices;
   offset_DPtrs  = NANOS_ALIGNED_MEMORY_OFFSET(offset_Data, size_Data, __alignof__( DD*) );

   // Computing Copies info
   num_Copies = wd->getNumCopies();
   num_Dimensions = 0;
   for ( i = 0; i < num_Copies; i += 1 ) {
      num_Dimensions += wd->getCopies()[i].getNumDimensions();
   }
   if ( num_Copies != 0 ) {
      size_CopyData = sizeof(CopyData);
      size_Copies   = size_CopyData * num_Copies;
      offset_Copies = NANOS_ALIGNED_MEMORY_OFFSET(offset_DPtrs, size_DPtrs, __alignof__(nanos_copy_data_t) );
      // There must be at least 1 dimension entry
      size_Dimensions = num_Dimensions * sizeof(nanos_region_dimension_internal_t);
      offset_Dimensions = NANOS_ALIGNED_MEMORY_OFFSET(offset_Copies, size_Copies, __alignof__(nanos_region_dimension_internal_t) );
   } else {
      size_Copies = 0;
      // No dimensions
      size_Dimensions = 0;
      offset_Copies = offset_Dimensions = NANOS_ALIGNED_MEMORY_OFFSET(offset_DPtrs, size_DPtrs, 1);
   }

   // Computing Internal Data info and total size
   static size_t size_PMD   = _pmInterface->getInternalDataSize();
   if ( size_PMD != 0 ) {
      static size_t align_PMD = _pmInterface->getInternalDataAlignment();
      offset_PMD = NANOS_ALIGNED_MEMORY_OFFSET(offset_Dimensions, size_Dimensions, align_PMD);
      total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_PMD,size_PMD,1);
   } else {
      offset_PMD = 0; // needed for a gcc warning
      total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_Dimensions, size_Dimensions, 1);
   }

   chunk = NEW char[total_size];

   // allocating WD and DATA; if size_Data == 0 data keep the NULL value
   if ( *uwd == NULL ) *uwd = (WD *) chunk;
   if ( size_Data != 0 ) {
      data = chunk + offset_Data;
      memcpy ( data, wd->getData(), size_Data );
   }

   // allocating Device Data
   DD **dev_ptrs = ( DD ** ) (chunk + offset_DPtrs);
   for ( i = 0 ; i < num_Devices; i ++ ) {
      dev_ptrs[i] = dev_data[i]->clone();
   }

   // allocate copy-in/copy-outs
   CopyData *wdCopies = ( CopyData * ) (chunk + offset_Copies);
   chunk_iter = chunk + offset_Copies;
   nanos_region_dimension_internal_t *dimensions = ( nanos_region_dimension_internal_t * ) ( chunk + offset_Dimensions );
   for ( i = 0; i < num_Copies; i++ ) {
      CopyData *wdCopiesCurr = ( CopyData * ) chunk_iter;
      *wdCopiesCurr = wd->getCopies()[i];
      memcpy( dimensions, wd->getCopies()[i].getDimensions(), sizeof( nanos_region_dimension_internal_t ) * wd->getCopies()[i].getNumDimensions() );
      wdCopiesCurr->setDimensions( dimensions );
      dimensions += wd->getCopies()[i].getNumDimensions();
      chunk_iter += size_CopyData;
   }

   // creating new WD 
   //FIXME jbueno (#758) should we have to take into account dimensions?
   new (*uwd) WD( *wd, dev_ptrs, wdCopies, data );

   // Set total size
   (*uwd)->setTotalSize(total_size );
   
   // initializing internal data
   if ( size_PMD != 0) {
      _pmInterface->initInternalData( chunk + offset_PMD );
      (*uwd)->setInternalData( chunk + offset_PMD );
      memcpy ( chunk + offset_PMD, wd->getInternalData(), size_PMD );
   }
}

/*! \brief Duplicates a given SlicedWD
 *
 *  This function duplicates the given as a parameter WD copying all the
 *  related data (devices ptr, data and DD)
 *
 *  \param [out] uwd is the target addr for the new WD
 *  \param [in] wd is the former WD
 */
void System::duplicateSlicedWD ( SlicedWD **uwd, SlicedWD *wd)
{
   unsigned int i, num_Devices, num_Copies;
   DeviceData **dev_data;
   void *data = NULL;
   char *chunk = 0, *chunk_iter;

   size_t size_CopyData;
   size_t size_Data, offset_Data, size_DPtrs, offset_DPtrs;
   size_t size_Copies, offset_Copies, size_PMD, offset_PMD;
   size_t total_size;

   // WD doesn't need to compute offset, it will always be the chunk allocated address

   // Computing Data info
   size_Data = wd->getDataSize();
   if ( *uwd == NULL ) offset_Data = NANOS_ALIGNED_MEMORY_OFFSET(0, sizeof(SlicedWD), wd->getDataAlignment() );
   else offset_Data = 0; // if there are no wd allocated, it will always be the chunk allocated address

   // Computing Data Device pointers and Data Devicesinfo
   num_Devices = wd->getNumDevices();
   dev_data = wd->getDevices();
   size_DPtrs    = sizeof(DD *) * num_Devices;
   offset_DPtrs  = NANOS_ALIGNED_MEMORY_OFFSET(offset_Data, size_Data, __alignof__( DD*) );

   // Computing Copies info
   num_Copies = wd->getNumCopies();
   if ( num_Copies != 0 ) {
      size_CopyData = sizeof(CopyData);
      size_Copies   = size_CopyData * num_Copies;
      offset_Copies = NANOS_ALIGNED_MEMORY_OFFSET(offset_DPtrs, size_DPtrs, __alignof__(nanos_copy_data_t) );
   } else {
      size_Copies = 0;
      offset_Copies = NANOS_ALIGNED_MEMORY_OFFSET(offset_DPtrs, size_DPtrs, 1);
   }

   // Computing Internal Data info and total size
   size_PMD   = _pmInterface->getInternalDataSize();
   if ( size_PMD != 0 ) {
      offset_PMD = NANOS_ALIGNED_MEMORY_OFFSET(offset_Copies, size_Copies, _pmInterface->getInternalDataAlignment());
   } else {
      offset_PMD = NANOS_ALIGNED_MEMORY_OFFSET(offset_Copies, size_Copies, 1);
   }

   total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_PMD, size_PMD, 1);

   chunk = NEW char[total_size];

   // allocating WD and DATA
   if ( *uwd == NULL ) *uwd = (SlicedWD *) chunk;
   if ( size_Data != 0 ) {
      data = chunk + offset_Data;
      memcpy ( data, wd->getData(), size_Data );
   }

   // allocating Device Data
   DD **dev_ptrs = ( DD ** ) (chunk + offset_DPtrs);
   for ( i = 0 ; i < num_Devices; i ++ ) {
      dev_ptrs[i] = dev_data[i]->clone();
   }

   // allocate copy-in/copy-outs
   CopyData *wdCopies = ( CopyData * ) (chunk + offset_Copies);
   chunk_iter = (chunk + offset_Copies);
   for ( i = 0; i < num_Copies; i++ ) {
      CopyData *wdCopiesCurr = ( CopyData * ) chunk_iter;
      *wdCopiesCurr = wd->getCopies()[i];
      chunk_iter += size_CopyData;
   }

   // creating new SlicedWD 
   new (*uwd) SlicedWD( *(wd->getSlicer()), *((WD *)wd), dev_ptrs, wdCopies, data );

   // Set total size
   (*uwd)->setTotalSize(total_size );
   
   // initializing internal data
   if ( size_PMD != 0) {
      _pmInterface->initInternalData( chunk + offset_PMD );
      (*uwd)->setInternalData( chunk + offset_PMD );
      memcpy ( chunk + offset_PMD, wd->getInternalData(), size_PMD );
   }
}

void System::setupWD ( WD &work, WD *parent )
{
   work.setParent ( parent );
   work.setDepth( parent->getDepth() +1 );
   
   // Inherit priority
   if ( parent != NULL ){
      // Add the specified priority to its parent's
      work.setPriority( work.getPriority() + parent->getPriority() );
   }

   // Prepare private copy structures to use relative addresses
   work.prepareCopies();

   // Invoke pmInterface
   _pmInterface->setupWD(work);
   Scheduler::updateCreateStats(work);
}

void System::submit ( WD &work )
{
   SchedulePolicy* policy = getDefaultSchedulePolicy();
   policy->onSystemSubmit( work, SchedulePolicy::SYS_SUBMIT );
   work.submit();
}

/*! \brief Submit WorkDescriptor to its parent's  dependencies domain
 */
void System::submitWithDependencies (WD& work, size_t numDataAccesses, DataAccess* dataAccesses)
{
   SchedulePolicy* policy = getDefaultSchedulePolicy();
   policy->onSystemSubmit( work, SchedulePolicy::SYS_SUBMIT_WITH_DEPENDENCIES );
   WD *current = myThread->getCurrentWD(); 
   current->submitWithDependencies( work, numDataAccesses , dataAccesses);
}

/*! \brief Wait on the current WorkDescriptor's domain for some dependenices to be satisfied
 */
void System::waitOn( size_t numDataAccesses, DataAccess* dataAccesses )
{
   WD* current = myThread->getCurrentWD();
   current->waitOn( numDataAccesses, dataAccesses );
}


void System::inlineWork ( WD &work )
{
   SchedulePolicy* policy = getDefaultSchedulePolicy();
   policy->onSystemSubmit( work, SchedulePolicy::SYS_INLINE_WORK );
   //! \todo choose actual (active) device...
   if ( Scheduler::checkBasicConstraints( work, *myThread ) ) {
      Scheduler::inlineWork( &work );
   } else {
      Scheduler::submitAndWait( work );
   }
}

void System::createWorker( unsigned p )
{
   NANOS_INSTRUMENT( sys.getInstrumentation()->incrementMaxThreads(); )
   PE *pe = createPE ( "smp", getBindingId( p ), _pes.size() );
   _pes.push_back ( pe );
   BaseThread *thread = &pe->startWorker();
   _workers.push_back( thread );
   ++_targetThreads;

   //Set up internal data
   WD & threadWD = thread->getThreadWD();
   if ( _pmInterface->getInternalDataSize() > 0 ) {
      char *data = NEW char[_pmInterface->getInternalDataSize()];
      _pmInterface->initInternalData( data );
      threadWD.setInternalData( data );
   }
   _pmInterface->setupWD( threadWD );
}

BaseThread * System::getUnassignedWorker ( void )
{
   BaseThread *thread;

   for ( unsigned i = 0; i < _workers.size(); i++ ) {
      thread = _workers[i];
      if ( !thread->hasTeam() || thread->isTaggedToSleep() ) {

         // skip if the thread is not in the mask
         if ( sys.getBinding() && !CPU_ISSET( thread->getCpuId(), &_cpu_active_set) )
            continue;

         // recheck availability with exclusive access
         thread->lock();

         if ( thread->hasTeam() && !thread->isTaggedToSleep() ) {
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

BaseThread * System:: getAssignedWorker ( void )
{
   BaseThread *thread;

   ThreadList::reverse_iterator rit;
   for ( rit = _workers.rbegin(); rit != _workers.rend(); ++rit ) {
      thread = *rit;
      if ( thread->hasTeam() && !thread->isTaggedToSleep() ) {

         // recheck availability with exclusive access
         thread->lock();

         if ( !thread->hasTeam() || thread->isTaggedToSleep() ) {
            // we lost it
            thread->unlock();
            continue;
         }

         thread->unlock();

         return thread;
      }
   }

   return NULL;
}

BaseThread * System::getWorker ( unsigned int n )
{
   if ( n < _workers.size() ) return _workers[n];
   else return NULL;
}

void System::acquireWorker ( ThreadTeam * team, BaseThread * thread, bool enter, bool star, bool creator )
{
   int thId = team->addThread( thread, star, creator );
   TeamData *data = NEW TeamData();

   data->setStar(star);

   SchedulePolicy &sched = team->getSchedulePolicy();
   ScheduleThreadData *sthdata = 0;
   if ( sched.getThreadDataSize() > 0 )
      sthdata = sched.createThreadData();

   data->setId(thId);
   data->setTeam(team);
   data->setScheduleData(sthdata);
   if ( creator )
      data->setParentTeamData(thread->getTeamData());

   if ( enter ) thread->enterTeam( data );
   else thread->setNextTeamData( data );

   // The sleep flag must be set before to avoid race conditions
   thread->wakeup();  // set sleep flag only
   thread->reserve(); // set team flag only

   debug( "added thread " << thread << " with id " << toString<int>(thId) << " to " << team );
}

void System::releaseWorker ( BaseThread * thread )
{
   ThreadTeam *team = thread->getTeam();
   unsigned thread_id = thread->getTeamId();

   //! \todo destroy if too many?
   debug("Releasing thread " << thread << " from team " << team );

   if ( _enable_dlb && thread->getTeamId() != 0 ) {
      // teamless threads will only sleep if DLB is enabled
      thread->sleep();
   } else {
      thread->leaveTeam();
      team->removeThread(thread_id);
   }
}

int System::getNumWorkers( DeviceData *arch )
{
   int n = 0;

   for ( ThreadList::iterator it = _workers.begin(); it != _workers.end(); it++ ) {
      if ( arch->isCompatible( ( *it )->runningOn()->getDeviceType() ) ) n++;
   }
   return n;
}

ThreadTeam * System::createTeam ( unsigned nthreads, void *constraints, bool reuseCurrent,
                                  bool enterCurrent, bool enterOthers, bool starringCurrent, bool starringOthers )
{
   if ( nthreads == 0 ) {
      nthreads = getNumThreads();
   }
   
   SchedulePolicy *sched = 0;
   if ( !sched ) sched = sys.getDefaultSchedulePolicy();

   ScheduleTeamData *stdata = 0;
   if ( sched->getTeamDataSize() > 0 )
      stdata = sched->createTeamData();

   // create team
   ThreadTeam * team = NEW ThreadTeam( nthreads, *sched, stdata, *_defBarrFactory(), *(_pmInterface->getThreadTeamData()),
                                       reuseCurrent ? myThread->getTeam() : NULL );

   debug( "Creating team " << team << " of " << nthreads << " threads" );

   // find threads
   if ( reuseCurrent ) {
      acquireWorker( team, myThread, enterCurrent, starringCurrent, /* creator */ true );
      nthreads--;
   }

   while ( nthreads > 0 ) {
      BaseThread *thread = getUnassignedWorker();

      if ( !thread ) {
         createWorker( _pes.size() );
         _numPEs++;
         _numThreads++;
         continue;
      }

      thread->lock();
      acquireWorker( team, thread, enterOthers, starringOthers, /* creator */ false );
      thread->signal();
      thread->unlock();
      nthreads--;
   }

   team->init();

   return team;
}

void System::endTeam ( ThreadTeam *team )
{
   debug("Destroying thread team " << team << " with size " << team->size() );

   dlb_returnCpusIfNeeded();
   while ( team->size ( ) > 0 ) {
      // FIXME: Is it really necessary?
      memoryFence();
   }
   
   fatal_cond( team->size() > 0, "Trying to end a team with running threads");
   
   delete team;
}

void System::updateActiveWorkers ( int nthreads )
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t num_threads_key = ID->getEventKey("set-num-threads"); )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &num_threads_key, (nanos_event_value_t *) &nthreads); )

   BaseThread *thread;
   ThreadTeam *team = myThread->getTeam();

   /* Increase _numThreads */
   while ( nthreads - _numThreads > 0 ) {

      thread = getUnassignedWorker();
      if ( !thread ) {
         createWorker( _pes.size() );
         _numPEs++;
         continue;
      }

      thread->lock();
      acquireWorker( team, thread, /* enterOthers */ true, /* starringOthers */ false, /* creator */ false );
      thread->signal();
      thread->unlock();
      _numThreads++;
   }

   /* Decrease _numThreads */
   while ( nthreads - _numThreads < 0 ) {
      thread = getAssignedWorker();
      thread->sleep();
      _numThreads--;
   }
}

// Not thread-safe
inline void System::applyCpuMask()
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t num_threads_key = ID->getEventKey("set-num-threads"); )
   NANOS_INSTRUMENT ( nanos_event_value_t num_threads_val = (nanos_event_value_t ) CPU_COUNT(&_cpu_active_set) )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &num_threads_key, &num_threads_val); )

   BaseThread *thread;
   ThreadTeam *team = myThread->getTeam();

   for ( unsigned pe_id = 0; pe_id < _pes.size() || _numPEs < (size_t)CPU_COUNT(&_cpu_active_set); pe_id++ ) {

      // Create PE & Worker if it does not exist
      if ( pe_id == _pes.size() ) {
         createWorker( pe_id );
      }

      bool pe_dirty = false;
      int pe_binding = getBindingId( pe_id );
      if ( CPU_ISSET( pe_binding, &_cpu_active_set) ) {

         // This PE should be running
         while ( (thread = _pes[pe_id]->getFirstStoppedThread()) != NULL ) {
            thread->lock();
            acquireWorker( team, thread, /* enterOthers */ true, /* starringOthers */ false, /* creator */ false );
            thread->signal();
            thread->unlock();
            _numThreads++;
            pe_dirty = true;
         }
         if ( pe_dirty ) _numPEs++;

      } else {

         // This PE should not
         while ( (thread = _pes[pe_id]->getFirstRunningThread()) != NULL ) {
            thread->sleep();
            _numThreads--;
            pe_dirty = true;
         }
         if ( pe_dirty ) _numPEs--;
      }
   }
   ensure( _numPEs ==  (size_t)CPU_COUNT(&_cpu_active_set), "applyCpuMask fatal error" );
}

void System::getCpuMask ( cpu_set_t *mask ) const
{
   memcpy( mask, &_cpu_active_set, sizeof(cpu_set_t) );
}

void System::setCpuMask ( const cpu_set_t *mask )
{
   memcpy( &_cpu_active_set, mask, sizeof(cpu_set_t) );
   sys.processCpuMask();
}

void System::addCpuMask ( const cpu_set_t *mask )
{
   CPU_OR( &_cpu_active_set, &_cpu_active_set, mask );
   sys.processCpuMask();
}

inline void System::processCpuMask( void )
{
   // if _bindThreads is enabled, update _bindings adding new elements of _cpu_active_set
   if ( sys.getBinding() ) {
      std::ostringstream oss_cpu_idx;
      oss_cpu_idx << "[";
      for ( int cpu=0; cpu<CPU_SETSIZE; cpu++) {
         if ( CPU_ISSET( cpu, &_cpu_active_set ) ) {

            if ( std::find( _bindings.begin(), _bindings.end(), cpu ) == _bindings.end() ) {
               _bindings.push_back( cpu );
            }

            oss_cpu_idx << cpu << ", ";
         }
      }
      oss_cpu_idx << "]";
      verbose0( "PID[" << getpid() << "]. CPU affinity " << oss_cpu_idx.str() );
      if ( _pmInterface->isMalleable() ) {
         sys.applyCpuMask();
      }
   }
   else {
      verbose0( "PID[" << getpid() << "]. Num threads: " << CPU_COUNT( &_cpu_active_set ) );
      if ( _pmInterface->isMalleable() ) {
         sys.updateActiveWorkers( CPU_COUNT( &_cpu_active_set ) );
      }
   }
}

void System::waitUntilThreadsPaused ()
{
   // Wait until all threads are paused
   _pausedThreadsCond.wait();
}

void System::waitUntilThreadsUnpaused ()
{
   // Wait until all threads are paused
   _unpausedThreadsCond.wait();
}

unsigned System::reservePE ( bool reserveNode, unsigned node, bool & reserved )
{
   // For each available PE
   for ( Bindings::reverse_iterator it = _bindings.rbegin(); it != _bindings.rend(); ++it )
   {
      unsigned pe = *it;
      unsigned currentNode = getNodeOfPE( pe );
      
      // If this PE is in the requested node or we don't need to reserve in
      // a certain node
      if ( currentNode == node || !reserveNode )
      {
         // Ensure there is at least one PE for smp
         if ( _bindings.size() == 1 )
         {
            reserved = false;
            warning( "Cannot reserve PE " << pe << ", there is just one PE left. It will be shared." );
         }
         else
         {
            // Take this pe out of the available bindings list.
            _bindings.erase( --( it.base() ) );
            reserved = true;
         }
         return pe;
      }
   }
   // If we reach this point, there are no PEs available for that node.
   verbose( "reservePE failed for node " << node );
   fatal( "There are no available PEs for the requested node" );
}

void * System::getHwlocTopology ()
{
   return _hwlocTopology;
}

void System::environmentSummary( void )
{
   /* Get Specific Mask String (depending on _bindThreads) */
   cpu_set_t *cpu_set = _bindThreads ? &_cpu_active_set : &_cpu_set;
   std::ostringstream mask;
   mask << "[ ";
   for ( int i=0; i<CPU_SETSIZE; i++ ) {
      if ( CPU_ISSET(i, cpu_set) )
         mask << i << ", ";
   }
   mask << "]";

   /* Get Prog. Model string */
   std::string prog_model;
   switch ( getInitialMode() )
   {
      case POOL:
         prog_model = "OmpSs";
         break;
      case ONE_THREAD:
         prog_model = "OpenMP";
         break;
      default:
         prog_model = "Unknown";
         break;
   }

   message0( "========== Nanos++ Initial Environment Summary ==========" );
   message0( "=== PID:            " << getpid() );
   message0( "=== Num. threads:   " << _numThreads );
   message0( "=== Active CPUs:    " << mask.str() );
   message0( "=== Binding:        " << std::boolalpha << _bindThreads );
   message0( "=== Prog. Model:    " << prog_model );

   for ( ArchitecturePlugins::const_iterator it = _archs.begin();
        it != _archs.end(); ++it ) {

      // Temporarily hide SMP plugin because it has empty information
      if ( strcmp( (*it)->getName(), "SMP PE Plugin" ) == 0 )
         continue;

      message0( "=== Plugin:         " << (*it)->getName() );
      message0( "===  | Threads:     " << (*it)->getNumThreads() );
   }

   message0( "=========================================================" );

   // Get start time
   _summary_start_time = time(NULL);
}

void System::admitCurrentThread ( void )
{
   int pe_id = _pes.size();   

   //! \note Create a new PE and configure it
   PE *pe = createPE ( "smp", getBindingId( pe_id ), pe_id );
   pe->setNUMANode( getNodeOfPE( pe->getId() ) );
   _pes.push_back ( pe );

   //! \note Create a new Thread object and associate it to the current thread
   BaseThread *thread = &pe->associateThisThread ( /* untie */ true ) ;
   _workers.push_back( thread );

   //! \note Update current cpu active set mask
   CPU_SET( getBindingId( pe_id ), &_cpu_active_set );

   //! \note Getting Programming Model interface data
   WD &mainWD = *myThread->getCurrentWD();
   (void) mainWD.getDirectory(true); // FIXME: this may cause future problems
   if ( _pmInterface->getInternalDataSize() > 0 ) {
      char *data = NEW char[_pmInterface->getInternalDataSize()];
      _pmInterface->initInternalData( data );
      mainWD.setInternalData( data );
   }

   //! \note Include thread into main thread
   acquireWorker( _mainTeam, thread, /* enter */ true, /* starring */ false, /* creator */ false );
   
}

void System::expelCurrentThread ( void )
{
   int pe_id =  myThread->runningOn()->getUId();
   _pes.erase( _pes.begin() + pe_id );
   _workers.erase ( _workers.begin() + myThread->getId() );
}

void System::executionSummary( void )
{
   time_t seconds = time(NULL) -_summary_start_time;
   message0( "============ Nanos++ Final Execution Summary ============" );
   message0( "=== Application ended in " << seconds << " seconds" );
   message0( "=== " << getCreatedTasks() << " tasks have been executed" );
   message0( "=========================================================" );
}
