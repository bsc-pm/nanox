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
#include <assert.h>
#include <string.h>
#include <signal.h>
#include <set>
#include <climits>
#include "smpthread.hpp"
#include "regiondict.hpp"
#include "smpprocessor.hpp"

#ifdef SPU_DEV
#include "spuprocessor.hpp"
#endif

#ifdef GPU_DEV
#include "gpuprocessor_decl.hpp"
#include "gpumemoryspace_decl.hpp"
#include "gpudd.hpp"
#endif

#ifdef CLUSTER_DEV
#include "clusternode_decl.hpp"
#include "clusterthread_decl.hpp"
#endif

#include "addressspace.hpp"

#ifdef OpenCL_DEV
#include "openclprocessor.hpp"
#endif

using namespace nanos;

System nanos::sys;

// default system values go here
System::System () :
      _atomicWDSeed( 1 ), _threadIdSeed( 0 ), _peIdSeed( 0 ),
      /*jb _numPEs( INT_MAX ), _numThreads( 0 ),*/ _deviceStackSize( 0 ), _profile( false ),
      _instrument( false ), _verboseMode( false ), _summary( false ), _executionMode( DEDICATED ), _initialMode( POOL ),
      _untieMaster( true ), _delayedStart( false ), _synchronizedStart( true ),
      _enableDLB( false ), _throttlePolicy ( NULL ),
      _schedStats(), _schedConf(), _defSchedule( "bf" ), _defThrottlePolicy( "hysteresis" ), 
      _defBarr( "centralized" ), _defInstr ( "empty_trace" ), _defDepsManager( "plain" ), _defArch( "smp" ),
      _initializedThreads ( 0 ), /*_targetThreads ( 0 ),*/ _pausedThreads( 0 ),
      _pausedThreadsCond(), _unpausedThreadsCond(),
      _net(), _usingCluster( false ), _usingNode2Node( true ), _usingPacking( true ), _conduit( "udp" ),
      _instrumentation ( NULL ), _defSchedulePolicy( NULL ), _dependenciesManager( NULL ),
      _pmInterface( NULL ), _masterGpuThd( NULL ), _separateMemorySpacesCount(1), _separateAddressSpaces(1024), _hostMemory( ext::SMP ),
      _regionCachePolicy( RegionCache::WRITE_BACK ), _regionCachePolicyStr(""), _clusterNodes(), _numaNodes(), _acceleratorCount(0)
#ifdef GPU_DEV
      , _pinnedMemoryCUDA( NEW CUDAPinnedMemoryManager() )
#endif
#ifdef NANOS_INSTRUMENTATION_ENABLED
      , _enableEvents(), _disableEvents(), _instrumentDefault("default"), _enableCpuidEvent( false )
#endif
      , _lockPoolSize(37), _lockPool( NULL ), _mainTeam (NULL), _simulator(false),  _task_max_retries(1), _affinityFailureCount( 0 )
      , _createLocalTasks( false )
      , _verboseDevOps( false )
      , _verboseCopies( false )
      , _splitOutputForThreads( false )
      , _userDefinedNUMANode( -1 )
      , _hwloc()
{
   verbose0 ( "NANOS++ initializing... start" );

   // OS::init must be called here and not in System::start() as it can be too late
   // to locate the program arguments at that point
   OS::init();
   config();

   _lockPool = NEW Lock[_lockPoolSize];

   if ( !_delayedStart ) {
      //std::cerr << "NX_ARGS is:" << (char *)(OS::getEnvironmentVariable( "NX_ARGS" ) != NULL ? OS::getEnvironmentVariable( "NX_ARGS" ) : "NO NX_ARGS: GG!") << std::endl;
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
   
#ifdef MPI_DEV
   char* isOffloadSlave = getenv(const_cast<char*> ("OMPSS_OFFLOAD_SLAVE")); 
   //Plugin->init of MPI will initialize MPI when we are slaves so MPI spawn returns ASAP in the master
   //This plugin does not reserve any PE at initialization time, just perform MPI Init and other actions
   if ( isOffloadSlave ) sys.loadPlugin("arch-mpi");
#endif
   
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

#ifdef CLUSTER_DEV
   if ( usingCluster() )
   {
      verbose0( "Loading Cluster plugin (" + getNetworkConduit() + ")" ) ;
      if ( !loadPlugin( "pe-cluster-"+getNetworkConduit() ) )
         fatal0 ( "Couldn't load Cluster support" );
   }
#endif

   verbose0( "Architectures loaded, stating PM interface.");
   _pmInterface->start();

   if ( !loadPlugin( "instrumentation-"+getDefaultInstrumentation() ) )
      fatal0( "Could not load " + getDefaultInstrumentation() + " instrumentation" );   

   // load default dependencies plugin
   verbose0( "loading " << getDefaultDependenciesManager() << " dependencies manager support" );

   if ( !loadPlugin( "deps-"+getDefaultDependenciesManager() ) )
      fatal0 ( "Couldn't load main dependencies manager" );

   ensure0( _dependenciesManager,"No default dependencies manager" );

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

   ensure0( _defBarrFactory,"No default system barrier factory" );
   

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

//   cfg.registerConfigOption( "num_threads", NEW Config::PositiveVar( _numThreads ),
//                             "Defines the number of threads. Note that OMP_NUM_THREADS is an alias to this." );
//   cfg.registerArgOption( "num_threads", "threads" );
//   cfg.registerEnvOption( "num_threads", "NX_THREADS" );
   
   cfg.registerConfigOption( "stack-size", NEW Config::PositiveVar( _deviceStackSize ),
                             "Defines the default stack size for all devices" );
   cfg.registerArgOption( "stack-size", "stack-size" );
   cfg.registerEnvOption( "stack-size", "NX_STACK_SIZE" );

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

   cfg.registerConfigOption( "instrument-cpuid", NEW Config::FlagOption ( _enableCpuidEvent ),
                             "Add cpuid event when binding is disabled (expensive)" );
   cfg.registerArgOption( "instrument-cpuid", "instrument-cpuid" );
#endif

   cfg.registerConfigOption( "enable-dlb", NEW Config::FlagOption ( _enableDLB ),
                              "Tune Nanos Runtime to be used with Dynamic Load Balancing library)" );
   cfg.registerArgOption( "enable-dlb", "enable-dlb" );

   /* Cluster: load the cluster support */
   cfg.registerConfigOption ( "enable-cluster", NEW Config::FlagOption ( _usingCluster, true ), "Enables the usage of Nanos++ Cluster" );
   cfg.registerArgOption ( "enable-cluster", "cluster" );
   //cfg.registerEnvOption ( "enable-cluster", "NX_ENABLE_CLUSTER" );

   cfg.registerConfigOption ( "no-node2node", NEW Config::FlagOption ( _usingNode2Node, false ), "Disables the usage of Slave-to-Slave transfers" );
   cfg.registerArgOption ( "no-node2node", "disable-node2node" );
   cfg.registerConfigOption ( "no-pack", NEW Config::FlagOption ( _usingPacking, false ), "Disables the usage of packing and unpacking of strided transfers" );
   cfg.registerArgOption ( "no-pack", "disable-packed-copies" );

   /* Cluster: select wich module to load mpi or udp */
   cfg.registerConfigOption ( "conduit", NEW Config::StringVar ( _conduit ), "Selects which GasNet conduit will be used" );
   cfg.registerArgOption ( "conduit", "cluster-network" );
   cfg.registerEnvOption ( "conduit", "NX_CLUSTER_NETWORK" );

   cfg.registerConfigOption ( "device-priority", NEW Config::StringVar ( _defDeviceName ), "Defines the default device to use");
   cfg.registerArgOption ( "device-priority", "--use-device");
   cfg.registerEnvOption ( "device-priority", "NX_USE_DEVICE");
   cfg.registerConfigOption( "simulator", NEW Config::FlagOption ( _simulator ),
                             "Nanos++ will be executed by a simulator (disabled as default)" );
   cfg.registerArgOption( "simulator", "simulator" );

   cfg.registerConfigOption( "task_retries", NEW Config::PositiveVar( _task_max_retries ),
                             "Defines the number of times a restartable task can be re-executed (default: 1). ");
   cfg.registerArgOption( "task_retries", "task-retries" );
   cfg.registerEnvOption( "task_retries", "NX_TASK_RETRIES" );


   cfg.registerConfigOption ( "verbose-devops", NEW Config::FlagOption ( _verboseDevOps, true ), "Verbose cache ops" );
   cfg.registerArgOption ( "verbose-devops", "verbose-devops" );
   cfg.registerConfigOption ( "verbose-copies", NEW Config::FlagOption ( _verboseCopies, true ), "Verbose data copies" );
   cfg.registerArgOption ( "verbose-copies", "verbose-copies" );

   cfg.registerConfigOption ( "thd-output", NEW Config::FlagOption ( _splitOutputForThreads, true ), "Create separate files for each thread" );
   cfg.registerArgOption ( "thd-output", "thd-output" );

   cfg.registerConfigOption ( "regioncache-policy", NEW Config::StringVar ( _regionCachePolicyStr ), "Region cache policy, accepted values are : nocache, writethrough, writeback. Default is writeback." );
   cfg.registerArgOption ( "regioncache-policy", "cache-policy" );
   cfg.registerEnvOption ( "regioncache-policy", "NX_CACHE_POLICY" );

   _schedConf.config( cfg );
   _pmInterface->config( cfg );
   
   _hwloc.config( cfg );

   verbose0 ( "Reading Configuration" );

   cfg.init();
}

void System::start ()
{
   _hwloc.loadHwloc();
   
   // Modules can be loaded now
   loadModules();

   // Instrumentation startup
   NANOS_INSTRUMENT ( sys.getInstrumentation()->filterEvents( _instrumentDefault, _enableEvents, _disableEvents ) );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->initialize() );

   verbose0 ( "Starting runtime" );

   if ( _regionCachePolicyStr.compare("") != 0 ) {
      //value is set
      if ( _regionCachePolicyStr.compare("nocache") == 0 ) {
         _regionCachePolicy = RegionCache::NO_CACHE;
      } else if ( _regionCachePolicyStr.compare("writethrough") == 0 ) {
         _regionCachePolicy = RegionCache::WRITE_THROUGH;
      } else if ( _regionCachePolicyStr.compare("writeback") == 0 ) {
         _regionCachePolicy = RegionCache::WRITE_BACK;
      } else {
         warning0("Invalid option for region cache policy '" << _regionCachePolicyStr << "', using default value.");
      }
   }

   _smpPlugin->associateThisThread( getUntieMaster() );
   //Setup MainWD
   WD &mainWD = *myThread->getCurrentWD();
   mainWD._mcontrol.setMainWD();

   if ( _pmInterface->getInternalDataSize() > 0 ) {
      char *data = NEW char[_pmInterface->getInternalDataSize()];
      _pmInterface->initInternalData( data );
      mainWD.setInternalData( data );
   }
   _pmInterface->setupWD( mainWD );

   /* Renaming currend thread as Master */
   myThread->rename("Master");
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent (NANOS_STARTUP) );

   for ( ArchitecturePlugins::const_iterator it = _archs.begin();
        it != _archs.end(); ++it )
   {
      verbose0("addPEs for arch: " << (*it)->getName()); 
      (*it)->addPEs( _pes );
      (*it)->addDevices( _devices );
   }

   
   
   for ( PEList::iterator it = _pes.begin(); it != _pes.end(); it++ ) {
      _clusterNodes.insert( it->second->getClusterNode() );
      if ( it->second->isInNumaNode() ) {
         // Add the node of this PE to the set of used NUMA nodes
         unsigned node = it->second->getNumaNode() ;
         _numaNodes.insert( node );
      }
   }
   
   // gmiranda: was completeNUMAInfo() We must do this after the
   // previous loop since we need the size of _numaNodes
   
   unsigned availNUMANodes = 0;
   // #994: this should be the number of NUMA objects in hwloc, but if we don't
   // want to query, this max should be enough
   unsigned maxNUMANode = _numaNodes.empty() ? 1 : *std::max_element( _numaNodes.begin(), _numaNodes.end() );
   // Create the NUMA node translation table. Do this before creating the team,
   // as the schedulers might need the information.
   _numaNodeMap.resize( maxNUMANode + 1, INT_MIN );
   
   for ( std::set<unsigned int>::const_iterator it = _numaNodes.begin();
        it != _numaNodes.end(); ++it )
   {
      unsigned node = *it;
      // If that node has not been translated, yet
      if ( _numaNodeMap[ node ] == INT_MIN )
      {
         verbose0( "[NUMA] Mapping from physical node " << node << " to user node " << availNUMANodes );
         _numaNodeMap[ node ] = availNUMANodes;
         // Increase the number of available NUMA nodes
         ++availNUMANodes;
      }
      // Otherwise, do nothing
   }
   verbose0( "[NUMA] " << availNUMANodes << " NUMA node(s) available for the user." );

   for ( ArchitecturePlugins::const_iterator it = _archs.begin();
        it != _archs.end(); ++it )
   {
      (*it)->startSupportThreads();
   }   
   
   for ( ArchitecturePlugins::const_iterator it = _archs.begin();
        it != _archs.end(); ++it )
   {
      (*it)->startWorkerThreads( _workers );
   }   

   // For each plugin, notify it's the way to reserve PEs if they are required
   //for ( ArchitecturePlugins::const_iterator it = _archs.begin();
   //     it != _archs.end(); ++it )
   //{
   //   (*it)->createBindingList();
   //}   

   _targetThreads = _smpPlugin->getNumThreads();

   // Set up internal data for each worker
   for ( ThreadList::const_iterator it = _workers.begin(); it != _workers.end(); it++ ) {

      WD & threadWD = it->second->getThreadWD();
      if ( _pmInterface->getInternalDataSize() > 0 ) {
         char *data = NEW char[_pmInterface->getInternalDataSize()];
         _pmInterface->initInternalData( data );
         threadWD.setInternalData( data );
      }
      _pmInterface->setupWD( threadWD );

      int schedDataSize = _defSchedulePolicy->getWDDataSize();
      if ( schedDataSize  > 0 ) {
         ScheduleWDData *schedData = reinterpret_cast<ScheduleWDData*>( NEW char[schedDataSize] );
         _defSchedulePolicy->initWDData( schedData );
         threadWD.setSchedulerData( schedData, true );
      }

   }

   if ( !_defDeviceName.empty() ) 
   {
       PEList::iterator it;
       for ( it = _pes.begin() ; it != _pes.end(); it++ )
       {
           PE *pe = it->second;
           if ( pe->getDeviceType()->getName() != NULL)
              if ( _defDeviceName == pe->getDeviceType()->getName()  )
                 _defDevice = pe->getDeviceType();
       }
   }

#ifdef NANOS_RESILIENCY_ENABLED
   // Setup signal handlers
   myThread->setupSignalHandlers();
#endif

   if ( getSynchronizedStart() )
      threadReady();


   switch ( getInitialMode() )
   {
      case POOL:
         verbose0("Pool model enabled (OmpSs)");
         _mainTeam = createTeam( _workers.size(), /*constraints*/ NULL, /*reuse*/ true, /*enter*/ true, /*parallel*/ false );
         break;
      case ONE_THREAD:
         verbose0("One-thread model enabled (OpenMP)");
         _mainTeam = createTeam( 1, /*constraints*/ NULL, /*reuse*/ true, /*enter*/ true, /*parallel*/ true );
         break;
      default:
         fatal("Unknown initial mode!");
         break;
   }

   if ( usingCluster() )
   {
      _net.nodeBarrier();
   }

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t num_threads_key = ID->getEventKey("set-num-threads"); )
   NANOS_INSTRUMENT ( nanos_event_value_t team_size =  (nanos_event_value_t) myThread->getTeam()->size(); )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &num_threads_key, &team_size); )
   
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
      
   if ( _summary )
      environmentSummary();
}

System::~System ()
{
   if ( !_delayedStart ) finish();
}

void System::finish ()
{
   //! \note Instrumentation: first removing RUNNING state from top of the state statck
   //! and then pushing SHUTDOWN state in order to instrument this latest phase
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent(NANOS_SHUTDOWN) );

   verbose ( "NANOS++ shutting down.... init" );

   //! \note waiting for remaining tasks
   myThread->getCurrentWD()->waitCompletion( true );

   //! \note switching main work descriptor (current) to the main thread to shutdown the runtime 
   if ( _workers[0]->isSleeping() ) _workers[0]->wakeup();
   getMyThreadSafe()->getCurrentWD()->tied().tieTo(*_workers[0]);
   Scheduler::switchToThread(_workers[0]);
   myThread->getTeam()->getSchedulePolicy().atShutdown();
   
   ensure( getMyThreadSafe()->isMainThread(), "Main thread not finishing the application!");

   //! \note stopping all threads
   verbose ( "Joining threads..." );
   for ( PEList::iterator it = _pes.begin(); it != _pes.end(); it++ ) {
      it->second->stopAllThreads();
   }
   verbose ( "...thread has been joined" );

   ensure( _schedStats._readyTasks == 0, "Ready task counter has an invalid value!");

   verbose ( "NANOS++ statistics");
   verbose ( std::dec << (unsigned int) getCreatedTasks() << " tasks has been executed" );

   sys.getNetwork()->nodeBarrier();

   for ( unsigned int nodeCount = 0; nodeCount < sys.getNetwork()->getNumNodes(); nodeCount += 1 ) {
      if ( sys.getNetwork()->getNodeNum() == nodeCount ) {
         for ( ArchitecturePlugins::const_iterator it = _archs.begin(); it != _archs.end(); ++it )
         {
            (*it)->finalize();
         }
#ifdef CLUSTER_DEV
         if ( _net.getNodeNum() == 0 && usingCluster() ) {
            //message0("Master: Created " << createdWds << " WDs.");
            //message0("Master: Failed to correctly schedule " << sys.getAffinityFailureCount() << " WDs.");
            //int soft_inv = 0;
            //int hard_inv = 0;

            //#ifdef OpenCL_DEV
            //      if ( _opencls ) {
            //         soft_inv = 0;
            //         hard_inv = 0;
            //         for ( unsigned int idx = 1; idx < _opencls->size(); idx += 1 ) {
            //            soft_inv += _separateAddressSpaces[(*_opencls)[idx]->getMemorySpaceId()]->getSoftInvalidationCount();
            //            hard_inv += _separateAddressSpaces[(*_opencls)[idx]->getMemorySpaceId()]->getHardInvalidationCount();
            //            //max_execd_wds = max_execd_wds >= (*_nodes)[idx]->getExecutedWDs() ? max_execd_wds : (*_nodes)[idx]->getExecutedWDs();
            //            //message("Memory space " << idx <<  " has performed " << _separateAddressSpaces[idx]->getSoftInvalidationCount() << " soft invalidations." );
            //            //message("Memory space " << idx <<  " has performed " << _separateAddressSpaces[idx]->getHardInvalidationCount() << " hard invalidations." );
            //         }
            //      }
            //      message0("OpenCLs Soft invalidations: " << soft_inv);
            //      message0("OpenCLs Hard invalidations: " << hard_inv);
            //#endif
         }
#endif
      }
      sys.getNetwork()->nodeBarrier();
   }

   //! \note finalizing instrumentation (if active)
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->finalize() );

   //! \note stopping and deleting the programming model interface
   _pmInterface->finish();
   delete _pmInterface;

   //! \note deleting pool of locks
   delete[] _lockPool;

   //! \note deleting main work descriptor
   delete ( WorkDescriptor * ) ( getMyThreadSafe()->getCurrentWD() );

   //! \note deleting loaded slicers
   for ( Slicers::const_iterator it = _slicers.begin(); it !=   _slicers.end(); it++ ) {
      delete ( Slicer * )  it->second;
   }

   //! \note deleting loaded worksharings
   for ( WorkSharings::const_iterator it = _worksharings.begin(); it !=   _worksharings.end(); it++ ) {
      delete ( WorkSharing * )  it->second;
   }
   
   //! \note  printing thread team statistics and deleting it
   ThreadTeam* team = getMyThreadSafe()->getTeam();

   if ( team->getScheduleData() != NULL ) team->getScheduleData()->printStats();

   myThread->leaveTeam();

   ensure(team->size() == 0, "Trying to finish execution, but team is still not empty");

   delete team;

   //! \note deleting processing elements (but main pe)
   for ( PEList::iterator it = _pes.begin(); it != _pes.end(); it++ ) {
      if ( it->first != (unsigned int)myThread->runningOn()->getId() ) {
         delete it->second;
      }
   }
   
   //! \note unload modules
   unloadModules();

   //! \note deleting dependency manager
   delete _dependenciesManager;

   //! \note deleting last processing element
   delete _pes[ myThread->runningOn()->getId() ];

   //! \note deleting allocator (if any)
   if ( allocator != NULL ) free (allocator);

   verbose0 ( "NANOS++ shutting down.... end" );
   //! \note printing execution summary
   if ( _summary ) executionSummary();

   _net.finalize(); //this can call exit (because of GASNet)
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
                        void **data, WD *uwg, nanos_wd_props_t *props, nanos_wd_dyn_props_t *dyn_props,
                        size_t num_copies, nanos_copy_data_t **copies, size_t num_dimensions,
                        nanos_region_dimension_internal_t **dimensions, nanos_translate_args_t translate_args,
                        const char *description, Slicer *slicer )
{
   ensure( num_devices > 0, "WorkDescriptor has no devices" );

   unsigned int i;
   char *chunk = 0;

   size_t size_CopyData;
   size_t size_Data, offset_Data, size_DPtrs, offset_DPtrs, size_Copies, offset_Copies, size_Dimensions, offset_Dimensions, offset_PMD;
   size_t offset_DESC, size_DESC;
   size_t offset_Sched;
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
   } else {
      offset_PMD = offset_DESC;
      size_PMD = size_DESC;
   }
   
   // Compute Scheduling Data size
   static size_t size_Sched = _defSchedulePolicy->getWDDataSize();
   if ( size_Sched != 0 )
   {
      static size_t align_Sched =  _defSchedulePolicy->getWDDataAlignment();
      offset_Sched = NANOS_ALIGNED_MEMORY_OFFSET(offset_PMD, size_PMD, align_Sched );
      total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_Sched,size_Sched,1);
   }
   else
   {
      offset_Sched = offset_PMD; // Needed by compiler unused variable error
      total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_PMD,size_PMD,1);
   }

   chunk = NEW char[total_size];
   if ( props != NULL ) {
      if (props->clear_chunk)
          memset(chunk, 0, sizeof(char) * total_size);
   }

   // allocating WD and DATA
   if ( *uwd == NULL ) *uwd = (WD *) chunk;
   if ( data != NULL && *data == NULL ) *data = (chunk + offset_Data);

   // allocating Device Data
   DD **dev_ptrs = ( DD ** ) (chunk + offset_DPtrs);
   for ( i = 0 ; i < num_devices ; i ++ ) dev_ptrs[i] = ( DD* ) devices[i].factory( devices[i].arg );

   //std::cerr << "num_copies=" << num_copies <<" copies=" <<copies << " num_dimensions=" <<num_dimensions << " dimensions=" << dimensions<< std::endl;
   //ensure ((num_copies==0 && copies==NULL && num_dimensions==0 && dimensions==NULL) || (num_copies!=0 && copies!=NULL && num_dimensions!=0 && dimensions!=NULL ), "Number of copies and copy data conflict" );
   ensure ((num_copies==0 && copies==NULL && num_dimensions==0 /*&& dimensions==NULL*/ ) || (num_copies!=0 && copies!=NULL && num_dimensions!=0 && dimensions!=NULL ), "Number of copies and copy data conflict" );
   

   // allocating copy-ins/copy-outs
   if ( copies != NULL && *copies == NULL ) {
      *copies = ( CopyData * ) (chunk + offset_Copies);
      ::bzero(*copies, size_Copies);
      *dimensions = ( nanos_region_dimension_internal_t * ) ( chunk + offset_Dimensions );
   }

   // Copying description string
   if ( description == NULL ) desc = NULL;
   else {
      desc = (chunk + offset_DESC);
      strncpy ( desc, description, size_DESC);
//      desc[strlen(description)]='\0';
   }

   WD * wd;
   wd =  new (*uwd) WD( num_devices, dev_ptrs, data_size, data_align, data != NULL ? *data : NULL,
                        num_copies, (copies != NULL)? *copies : NULL, translate_args, desc );

   if ( slicer ) wd->setSlicer(slicer);

   // Set WD's socket
   wd->setNUMANode( sys.getUserDefinedNUMANode() );
   
   // Set total size
   wd->setTotalSize(total_size );
   
   if ( wd->getNUMANode() >= (int)sys.getNumNumaNodes() )
      throw NANOS_INVALID_PARAM;

   // All the implementations for a given task will have the same ID
   wd->setVersionGroupId( ( unsigned long ) devices );

   // initializing internal data
   if ( size_PMD > 0) {
      _pmInterface->initInternalData( chunk + offset_PMD );
      wd->setInternalData( chunk + offset_PMD );
   }
   
   // Create Scheduling data
   if ( size_Sched > 0 ){
      _defSchedulePolicy->initWDData( chunk + offset_Sched );
      ScheduleWDData * sched_Data = reinterpret_cast<ScheduleWDData*>( chunk + offset_Sched );
      wd->setSchedulerData( sched_Data, /*ownedByWD*/ false );
   }

   // add to workdescriptor
   if ( uwg != NULL ) {
      WD * wg = ( WD * )uwg;
      wg->addWork( *wd );
   }

   // set properties
   if ( props != NULL ) {
      if ( props->tied ) wd->tied();
      wd->setPriority( dyn_props->priority );
      wd->setFinal ( dyn_props->flags.is_final );
      wd->setRecoverable ( dyn_props->flags.is_recover);
   }
   if ( dyn_props && dyn_props->tie_to ) wd->tieTo( *( BaseThread * )dyn_props->tie_to );
   
   /* DLB */
   // In case the master have been busy crating tasks 
   // every 10 tasks created I'll check available cpus
   if(_atomicWDSeed.value()%10==0)dlb_updateAvailableCpus();

   if (_createLocalTasks) {
      wd->tieToLocation( 0 );
   }
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
 *  \sa WorkDescriptor, createWD 
 */
void System::duplicateWD ( WD **uwd, WD *wd)
{
   unsigned int i, num_Devices, num_Copies, num_Dimensions;
   DeviceData **dev_data;
   void *data = NULL;
   char *chunk = 0, *chunk_iter;

   size_t size_CopyData;
   size_t size_Data, offset_Data, size_DPtrs, offset_DPtrs, size_Copies, offset_Copies, size_Dimensions, offset_Dimensions, offset_PMD;
   size_t offset_Sched;
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
   } else {
      offset_PMD = offset_Copies;
      size_PMD = size_Copies;
   }

   // Compute Scheduling Data size
   static size_t size_Sched = _defSchedulePolicy->getWDDataSize();
   if ( size_Sched != 0 )
   {
      static size_t align_Sched =  _defSchedulePolicy->getWDDataAlignment();
      offset_Sched = NANOS_ALIGNED_MEMORY_OFFSET(offset_PMD, size_PMD, align_Sched );
      total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_Sched,size_Sched,1);
   }
   else
   {
      offset_Sched = offset_PMD; // Needed by compiler unused variable error
      total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_PMD,size_PMD,1);
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
   
   // Create Scheduling data
   if ( size_Sched > 0 ){
      _defSchedulePolicy->initWDData( chunk + offset_Sched );
      ScheduleWDData * sched_Data = reinterpret_cast<ScheduleWDData*>( chunk + offset_Sched );
      (*uwd)->setSchedulerData( sched_Data, /*ownedByWD*/ false );
   }
}

void System::setupWD ( WD &work, WD *parent )
{
   work.setDepth( parent->getDepth() +1 );
   
   // Inherit priority
   if ( parent != NULL ){
      // Add the specified priority to its parent's
      work.setPriority( work.getPriority() + parent->getPriority() );
   }

   /**************************************************/
   /*********** selective node executuion ************/
   /**************************************************/
   //if (sys.getNetwork()->getNodeNum() == 0) work.tieTo(*_workers[ 1 + nanos::ext::GPUConfig::getGPUCount() + ( work.getId() % ( sys.getNetwork()->getNumNodes() - 1 ) ) ]);
   /**************************************************/
   /**************************************************/

   //  ext::SMPDD * workDD = dynamic_cast<ext::SMPDD *>( &work.getActiveDevice());
   //if (sys.getNetwork()->getNodeNum() == 0)
   //         std::cerr << "wd " << work.getId() << " depth is: " << work.getDepth() << " @func: " << (void *) workDD->getWorkFct() << std::endl;
#if 0
#ifdef CLUSTER_DEV
   if (sys.getNetwork()->getNodeNum() == 0)
   {
      //std::cerr << "tie wd " << work.getId() << " to my thread" << std::endl;
      //ext::SMPDD * workDD = dynamic_cast<ext::SMPDD *>( &work.getActiveDevice());
      switch ( work.getDepth() )
      {
         //case 1:
         //   //std::cerr << "tie wd " << work.getId() << " to my thread, @func: " << (void *) workDD->getWorkFct() << std::endl;
         //   work.tieTo( *myThread );
         //   break;
         //case 1:
            //if (work.canRunIn( ext::GPU) )
            //{
            //   work.tieTo( *_masterGpuThd );
            //}
         //   break;
         default:
            break;
            std::cerr << "wd " << work.getId() << " depth is: " << work.getDepth() << " @func: " << (void *) workDD->getWorkFct() << std::endl;
      }
   }
#endif
#endif
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

/*
   if (_net.getNodeNum() > 0 ) setupWD( work, getSlaveParentWD() );
   else setupWD( work, myThread->getCurrentWD() );
*/

   work.submit();
}

/*! \brief Submit WorkDescriptor to its parent's  dependencies domain
 */
void System::submitWithDependencies (WD& work, size_t numDataAccesses, DataAccess* dataAccesses)
{
   SchedulePolicy* policy = getDefaultSchedulePolicy();
   policy->onSystemSubmit( work, SchedulePolicy::SYS_SUBMIT_WITH_DEPENDENCIES );
/*
   setupWD( work, myThread->getCurrentWD() );
*/
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
      work._mcontrol.preInit();
      work._mcontrol.initialize( *( myThread->runningOn() ) );
      bool result;
      do {
         result = work._mcontrol.allocateTaskMemory();
      } while( result == false );
      Scheduler::inlineWork( &work );
   }
   else fatal ("System: Trying to execute inline a task violating basic constraints");
}

void System::createWorker( unsigned p )
{
   fatal0("Disabled");
   //jb NANOS_INSTRUMENT( sys.getInstrumentation()->incrementMaxThreads(); )
   //jb PE *pe = createPE ( "smp", getBindingId( p ), _pes.size() );
   //jb _pes.push_back ( pe );
   //jb BaseThread *thread = &pe->startWorker();
   //jb _workers.push_back( thread );
   //jb ++_targetThreads;

   //jb CPU_SET( getBindingId( p ), &_smpPlugin->getActiveSet() );

   //jb //Set up internal data
   //jb WD & threadWD = thread->getThreadWD();
   //jb if ( _pmInterface->getInternalDataSize() > 0 ) {
   //jb    char *data = NEW char[_pmInterface->getInternalDataSize()];
   //jb    _pmInterface->initInternalData( data );
   //jb    threadWD.setInternalData( data );
   //jb }
   //jb _pmInterface->setupWD( threadWD );
}

BaseThread * System::getUnassignedWorker ( void )
{
   BaseThread *thread;

   for ( ThreadList::iterator it = _workers.begin(); it != _workers.end(); it++ ) {
      thread = it->second;
      if ( !thread->hasTeam() && !thread->isSleeping() ) {

         // skip if the thread is not in the mask
         if ( _smpPlugin->getBinding() && !CPU_ISSET( thread->getCpuId(), &_smpPlugin->getActiveSet() ) ) {
            continue;
         }

         // recheck availability with exclusive access
         thread->lock();

         if ( thread->hasTeam() || thread->isSleeping()) {
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

BaseThread * System::getWorker ( unsigned int n )
{
   BaseThread *worker = NULL;
   ThreadList::iterator elem = _workers.find( n );
   if ( elem != _workers.end() ) {
      worker = elem->second;
   } 
   return worker;
}

void System::acquireWorker ( ThreadTeam * team, BaseThread * thread, bool enter, bool star, bool creator )
{
   int thId = team->addThread( thread, star, creator );
   TeamData *data = NEW TeamData();
   if ( creator ) data->setCreator( true );

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

   debug( "added thread " << thread << " with id " << toString<int>(thId) << " to " << team );
}

void System::releaseWorker ( BaseThread * thread )
{
   ensure( myThread == thread, "Calling release worker from other thread context" );

   //! \todo Destroy if too many?
   debug("Releasing thread " << thread << " from team " << thread->getTeam() );

   thread->lock();
   thread->sleep();
   thread->unlock();

}

int System::getNumWorkers( DeviceData *arch )
{
   int n = 0;

   for ( ThreadList::iterator it = _workers.begin(); it != _workers.end(); it++ ) {
      if ( arch->isCompatible( *(it->second->runningOn()->getDeviceType() ) ), it->second->runningOn() ) n++;
   }
   return n;
}

ThreadTeam * System::createTeam ( unsigned nthreads, void *constraints, bool reuse, bool enter, bool parallel )
{
   //! \note Getting default scheduler
   SchedulePolicy *sched = sys.getDefaultSchedulePolicy();

   //! \note Getting scheduler team data (if any)
   ScheduleTeamData *std = ( sched->getTeamDataSize() > 0 )? sched->createTeamData() : NULL;

   //! \note create team object
   ThreadTeam * team = NEW ThreadTeam( nthreads, *sched, std, *_defBarrFactory(), *(_pmInterface->getThreadTeamData()),
                                       reuse? myThread->getTeam() : NULL );

   debug( "Creating team " << team << " of " << nthreads << " threads" );

   team->setFinalSize(nthreads);

   //! \note Reusing current thread
   if ( reuse ) {
      acquireWorker( team, myThread, /* enter */ enter, /* staring */ true, /* creator */ true );
      nthreads--;
   }
   
   //! \note Getting rest of the members 
   while ( nthreads > 0 ) {

      BaseThread *thread = getUnassignedWorker();
      ensure( thread != NULL, "I could not get the required threads to create the team");

      acquireWorker( team, thread, /*enter*/ enter, /* staring */ parallel, /* creator */ false );

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
 
void System::addPEsAndThreadsToTeam(PE **pes, int num_pes, BaseThread** threads, int num_threads) {  
    //Insert PEs to the team
    for (int i=0; i<num_pes; i++){
        _pes.insert( std::make_pair( pes[i]->getId(), pes[i] ) );
    }
    //Insert the workers to the team
    for (int i=0; i<num_threads; i++){
        _workers.insert( std::make_pair( threads[i]->getId(), threads[i] ) );
        acquireWorker( _mainTeam , threads[i] );
    }
}

void System::admitCurrentThread ( bool isWorker )
{
   _smpPlugin->admitCurrentThread( _workers, isWorker );
}

void System::expelCurrentThread ( bool isWorker )
{
   _smpPlugin->expelCurrentThread( _workers, isWorker );
}

void System::environmentSummary( void )
{
   std::ostringstream mask;
   _smpPlugin->getBindingMaskString( mask );

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
   message0( "=== PID:                 " << getpid() );
   //message0( "=== Num. SMP threads:        " << _smpPlugin->getNumThreads() );
   //message0( "=== Num. SMP worker threads: " << _smpPlugin->getNumWorkers() );
   message0( "=== Num. worker threads: " << _workers.size() );
   message0( "=== System CPUs:         " << mask.str() );
   message0( "=== Binding:             " << std::boolalpha << _smpPlugin->getBinding() );
   message0( "=== Prog. Model:         " << prog_model );

   for ( ArchitecturePlugins::const_iterator it = _archs.begin();
        it != _archs.end(); ++it ) {
      message0( "=== Plugin:              " << (*it)->getName() );
      message0( "===  | PEs:              " << (*it)->getNumPEs() );
      message0( "===  | Threads:          " << (*it)->getNumThreads() );
      message0( "===  | Worker Threads:   " << (*it)->getNumWorkers() );
   }

   message0( "=========================================================" );

   // Get start time
   _summaryStartTime = time(NULL);
}

void System::executionSummary( void )
{
   time_t seconds = time(NULL) -_summaryStartTime;
   message0( "============ Nanos++ Final Execution Summary ============" );
   message0( "=== Application ended in " << seconds << " seconds" );
   message0( "=== " << getCreatedTasks() << " tasks have been executed" );
   message0( "=========================================================" );
}

//If someone needs argc and argv, it may be possible, but then a fortran 
//main should be done too
void System::ompss_nanox_main(){
    #ifdef MPI_DEV
    if (getenv("OMPSS_OFFLOAD_SLAVE")){
        //Plugin->init of MPI will do everything and then exit(0)
        sys.loadPlugin("arch-mpi");
    }
    #endif
    #ifdef CLUSTER_DEV
    nanos::ext::ClusterNode::clusterWorker();
    #endif
    
    #ifdef NANOS_RESILIENCY_ENABLED
        getMyThreadSafe()->setupSignalHandlers();
    #endif
}

void System::_registerMemoryChunk(memory_space_id_t loc, void *addr, std::size_t len) {
   CopyData cd;
   nanos_region_dimension_internal_t dim;
   dim.lower_bound = 0;
   dim.size = len;
   dim.accessed_length = len;
   cd.setBaseAddress( addr );
   cd.setDimensions( &dim );
   cd.setNumDimensions( 1 );
   global_reg_t reg;
   getHostMemory().getRegionId( cd, reg, *((WD *) 0), 0 );
   reg.setOwnedMemory(loc);
   //not really needed.., *it->registerOwnedMemory( reg );
}

void System::registerNodeOwnedMemory(unsigned int node, void *addr, std::size_t len) {
   memory_space_id_t loc = 0;
   if ( node == 0 ) {
      _registerMemoryChunk( loc, addr, len );
   } else {
      //_separateAddressSpaces[0] is always NULL (because loc = 0 is the local node memory)
      for ( std::vector<SeparateMemoryAddressSpace *>::iterator it = _separateAddressSpaces.begin(); it != _separateAddressSpaces.end(); it++ ) {
         if ( *it != NULL ) {
            if ((*it)->getNodeNumber() == node) {
               _registerMemoryChunk( loc, addr, len );
            }
         }
         loc++;
      }
   }
}

void System::stickToProducer(void *addr, std::size_t len) {
   if ( _net.getNodeNum() == Network::MASTER_NODE_NUM ) {
      CopyData cd;
      nanos_region_dimension_internal_t dim;
      dim.lower_bound = 0;
      dim.size = len;
      dim.accessed_length = len;
      cd.setBaseAddress( addr );
      cd.setDimensions( &dim );
      cd.setNumDimensions( 1 );
      global_reg_t reg;
      getHostMemory().getRegionId( cd, reg, *((WD *) 0), 0 );
      reg.key->setKeepAtOrigin( true );
   }
}

void System::setCreateLocalTasks( bool value ) {
   _createLocalTasks = value;
}

memory_space_id_t System::addSeparateMemoryAddressSpace( Device &arch, bool allocWide ) {
   memory_space_id_t id = getNewSeparateMemoryAddressSpaceId();
   SeparateMemoryAddressSpace *mem = NEW SeparateMemoryAddressSpace( id, arch, allocWide );
   _separateAddressSpaces[ id ] = mem;
   return id;
}

void System::registerObject(int numObjects, nanos_copy_data_internal_t *obj) {
   for ( int i = 0; i < numObjects; i += 1 ) {
      _hostMemory.registerObject( &obj[i] );
   }
}
