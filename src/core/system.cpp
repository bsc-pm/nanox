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

#include <string.h>
#include "system.hpp"
#include "config.hpp"
#include "plugin.hpp"
#include "schedule.hpp"
#include "barrier.hpp"
#include "nanos-int.h"
#include "copydata.hpp"
#include "os.hpp"

#ifdef SPU_DEV
#include "spuprocessor.hpp"
#endif

#ifdef GPU_DEV
#include "gpuprocessor_fwd.hpp"
#endif

#ifdef CLUSTER_DEV
#include "clusternode.hpp"
#include "clusternodeinfo.hpp"
#endif

using namespace nanos;

namespace nanos {
  System::Init externInit __attribute__((weak));
}

System nanos::sys;

// default system values go here
System::System () :
      _numPEs( 1 ), _deviceStackSize( 0 ), _bindThreads( true ), _profile( false ), _instrument( false ),
      _verboseMode( false ), _executionMode( DEDICATED ), _initialMode(POOL), _thsPerPE( 1 ), _untieMaster( true ), _delayedStart(false), _isMaster(true), _defSchedule( "bf" ), _defThrottlePolicy( "numtasks" ), _defBarr( "posix" ),
      _defInstr ( "empty_trace" ), _defArch("smp"), _instrumentor ( NULL ), _defSchedulePolicy(NULL), _directory()
{
   verbose0 ( "NANOS++ initalizing... start" );
   // OS::init must be called here and not in System::start() as it can be too late
   // to locate the program arguments at that point
   OS::init();
   config();
   if ( !_delayedStart ) {
      start();
   }
   verbose0 ( "NANOS++ initalizing... end" );
}

void System::loadModules ()
{
   verbose0 ( "Configuring module manager" );

   PluginManager::init();

   verbose0 ( "Loading modules" );

   // load host processor module
   verbose0( "loading SMP support" );

   if ( !PluginManager::load ( "pe-"+getDefaultArch() ) )
      fatal0 ( "Couldn't load host support" );

   ensure( _hostFactory,"No default host factory" );

#ifdef GPU_DEV
   verbose0( "loading GPU support" );

   if ( !PluginManager::load ( "pe-gpu" ) )
      fatal0 ( "Couldn't load GPU support" );
#endif

#ifdef CLUSTER_DEV
   if ( !PluginManager::load ( "pe-cluster" ) )
      fatal0 ( "Couldn't load Cluster support" );
   fprintf(stderr, "Cluster plugin loaded!\n");
#endif

   // load default schedule plugin
   verbose0( "loading " << getDefaultSchedule() << " scheduling policy support" );

   if ( !PluginManager::load ( "sched-"+getDefaultSchedule() ) )
      fatal0 ( "Couldn't load main scheduling policy" );

   ensure( _defSchedulePolicy,"No default system scheduling factory" );

   verbose0( "loading " << getDefaultThrottlePolicy() << " throttle policy" );

   if ( !PluginManager::load( "throttle-"+getDefaultThrottlePolicy() ) )
      fatal0( "Could not load main cutoff policy" );

   ensure(_throttlePolicy, "No default throttle policy");

   verbose0( "loading " << getDefaultBarrier() << " barrier algorithm" );

   if ( !PluginManager::load( "barrier-"+getDefaultBarrier() ) )
      fatal0( "Could not load main barrier algorithm" );

   if ( !PluginManager::load( "instrumentation-"+getDefaultInstrumentor() ) )
      fatal0( "Could not load " + getDefaultInstrumentor() + " instrumentation" );


   ensure( _defBarrFactory,"No default system barrier factory" );

}


void System::config ()
{
   Config config;

   if ( externInit != NULL ) {
        externInit();
   }

   verbose0 ( "Preparing library configuration" );

   config.setOptionsSection ( "Core", "Core options of the core of Nanos++ runtime"  );

   config.registerConfigOption ( "num_pes", new Config::PositiveVar( _numPEs ), "Defines the number of processing elements" );
   config.registerArgOption ( "num_pes", "pes" );
   config.registerEnvOption ( "num_pes", "NX_PES" );

   config.registerConfigOption ( "stack-size", new Config::PositiveVar( _deviceStackSize ), "Defines the default stack size for all devices" );
   config.registerArgOption ( "stack-size", "stack-size" );
   config.registerEnvOption ( "stack-size", "NX_STACK_SIZE" );

   config.registerConfigOption ( "no-binding", new Config::FlagOption( _bindThreads, false), "Disables thread binding" );
   config.registerArgOption ( "no-binding", "disable-binding" );

   config.registerConfigOption ( "verbose", new Config::FlagOption( _verboseMode), "Activates verbose mode" );
   config.registerArgOption ( "verbose", "verbose" );

#if 0
   FIXME: implement execution modes (#146)
   Config::MapVar<ExecutionMode> map( _executionMode );
   map.addOption( "dedicated", DEDICATED).addOption( "shared", SHARED );
   config.registerConfigOption ( "exec_mode", &map, "Execution mode" );
   config.registerArgOption ( "exec_mode", "mode" );
#endif

   config.registerConfigOption ( "schedule", new Config::StringVar ( _defSchedule ), "Defines the scheduling policy" );
   config.registerArgOption ( "schedule", "schedule" );
   config.registerEnvOption ( "schedule", "NX_SCHEDULE" );

   config.registerConfigOption ( "throttle", new Config::StringVar ( _defThrottlePolicy ), "Defines the throttle policy" );
   config.registerArgOption ( "throttle", "throttle" );
   config.registerEnvOption ( "throttle", "NX_THROTTLE" );

   config.registerConfigOption ( "barrier", new Config::StringVar ( _defBarr ), "Defines barrier algorithm" );
   config.registerArgOption ( "barrier", "barrier" );
   config.registerEnvOption ( "barrier", "NX_BARRIER" );

   config.registerConfigOption ( "instrumentation", new Config::StringVar ( _defInstr ), "Defines instrumentation format" );
   config.registerArgOption ( "instrumentation", "instrumentation" );
   config.registerEnvOption ( "instrumentation", "NX_INSTRUMENTATION" );

   _schedConf.config(config);
   
   verbose0 ( "Reading Configuration" );
   config.init();
}

PE * System::createPE ( std::string pe_type, int pid )
{
   // TODO: lookup table for PE factories
   // in the mean time assume only one factory

   return _hostFactory( pid );
}

void System::start ()
{
   loadModules();

   // Instrumentation startup
   NANOS_INSTRUMENT ( sys.getInstrumentor()->initialize() );

   verbose0 ( "Starting threads" );

   int numPes = getNumPEs();

   _pes.reserve ( numPes );

   PE *pe = createPE ( "smp", 0 );
   _pes.push_back ( pe );
   _workers.push_back( &pe->associateThisThread ( getUntieMaster() ) );


   NANOS_INSTRUMENT ( sys.getInstrumentor()->raiseOpenStateEvent (STARTUP) );

   //start as much threads per pe as requested by the user
   for ( int ths = 1; ths < getThsPerPE(); ths++ ) {
      _workers.push_back( &pe->startWorker( ));
   }

   int p;
   for ( p = 1; p < numPes ; p++ ) {
      pe = createPE ( "smp", p );
      _pes.push_back ( pe );

      //starting as much threads per pe as requested by the user

      for ( int ths = 0; ths < getThsPerPE(); ths++ ) {
         _workers.push_back( &pe->startWorker() );
      }
   }

#ifdef GPU_DEV
   int gpuC;
   for ( gpuC = 0; gpuC < nanos::ext::GPUDD::getGPUCount(); gpuC++ ) {
      PE *gpu = new nanos::ext::GPUProcessor( p++, gpuC );
      _pes.push_back( gpu );
      _workers.push_back( &gpu->startWorker() );
   }
#endif

#ifdef SPU_DEV
   PE *spu = new nanos::ext::SPUProcessor(100, (nanos::ext::SMPProcessor &) *_pes[0]);
   spu->startWorker();
#endif

   _net.initialize();

#ifdef CLUSTER_DEV
   if ( _net.getNodeNum() == 0 )
   {
      unsigned int nodeC;
      for ( nodeC = 0; nodeC < _net.getNumNodes(); nodeC++ ) {
         if ( nodeC != _net.getNodeNum() ) {
            PE *node = new nanos::ext::ClusterRemoteNode( nodeC );
            _pes.push_back( node );
            _workers.push_back( &node->startWorker() );
         }
      }
   }
#endif

   switch ( getInitialMode() )
   {
      case POOL:
         createTeam( _workers.size() );
         break;
      case ONE_THREAD:
         createTeam(1);
         break;
      default:
         fatal("Unknown inital mode!");
         break;
   }
   NANOS_INSTRUMENT ( sys.getInstrumentor()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( sys.getInstrumentor()->raiseOpenStateEvent (RUNNING) );

#ifdef CLUSTER_DEV
   setMaster(_net.getNodeNum() == nanos::Network::MASTER_NODE_NUM);
   if (!isMaster())
   {
       Scheduler::workerLoop();
       fprintf(stderr, "Slave node: I have to finish.\n");
       finish();
   }
   //else
   //   fprintf(stderr, "Im only allowed here if im the master.\n");
#endif
}

System::~System ()
{
   if ( !_delayedStart ) finish();
}

void System::finish ()
{
   /* Instrumentor: First removing RUNNING state from top of the state statck */
   NANOS_INSTRUMENT ( sys.getInstrumentor()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( sys.getInstrumentor()->raiseOpenStateEvent(SHUTDOWN) );

   verbose ( "NANOS++ shutting down.... init" );
   verbose ( "Wait for main workgroup to complete" );
   myThread->getCurrentWD()->waitCompletion();

   // we need to switch to the main thread here to finish
   // the execution correctly
   myThread->getCurrentWD()->tieTo(*_workers[0]);
   Scheduler::switchToThread(_workers[0]);
   
   //FIXME (#185) : ensure(myThread->getId() == 0, "Main thread not finishing the application!");

   verbose ( "Joining threads... phase 1" );
   // signal stop PEs

//#ifdef CLUSTER_DEV
//   for ( unsigned p = 1; p < _pes.size() ; p++ ) {
//       if (_pes[p] != nanos::ext::ClusterNodeInfo::getThisNodePE())
//       {
//           ((nanos::ext::ClusterRemoteNode *) _pes[p])->stopAll();
//       }
//   }
//   nanos::ext::ClusterNodeInfo::callNetworkFinalizeFunc();
//   fprintf(stderr, "Closed the network.\n");
//#else
   fprintf(stderr, "Finishing thds.\n");
   for ( unsigned p = 1; p < _pes.size() ; p++ ) {
       _pes[p]->stopAll();
   }
//#endif

   verbose ( "Joining threads... phase 2" );


   // shutdown instrumentation
   NANOS_INSTRUMENT ( sys.getInstrumentor()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( sys.getInstrumentor()->finalize() );

   // join
   for ( unsigned p = 1; p < _pes.size() ; p++ ) {
      delete _pes[p];
   }
   verbose ( "NANOS++ shutting down.... end" );

   _net.getAPI()->finalize();
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
 *
 *  When it does a full allocation the layout is the following:
 *
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
 *  |    copyN      |
 *  +---------------+
 *
 */
void System::createWD ( WD **uwd, size_t num_devices, nanos_device_t *devices, size_t data_size,
                        void **data, WG *uwg, nanos_wd_props_t *props, size_t num_copies, nanos_copy_data_t **copies )
{
   int dd_size = 0;
   for ( unsigned int i = 0; i < num_devices; i++ )
      dd_size += devices[i].dd_size;

   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures (temporary solved)
   int size_to_allocate = ( ( *uwd == NULL ) ? sizeof( WD ) : 0 ) +
                          ( ( data != NULL && *data == NULL ) ? (((data_size+7)>>3)<<3) : 0 ) +
                          sizeof( DD* ) * num_devices +
                          dd_size +
                          ( ( copies != NULL && *copies == NULL ) ? num_copies * sizeof(CopyData) : 0 )
                          ;

   char *chunk = 0;

   if ( size_to_allocate ) chunk = new char[size_to_allocate];

   // allocate WD
   if ( *uwd == NULL ) {
      *uwd = ( WD * ) chunk;
      chunk += sizeof( WD );
   }

   // allocate WD data
   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures (temporary solved)
   if ( data != NULL && *data == NULL ) {
      *data = chunk;
      chunk += (((data_size+7)>>3)<<3);
   }

   // allocate device pointers vector
   DD **dev_ptrs = ( DD ** ) chunk;
   chunk += sizeof( DD* ) * num_devices;

   // allocate device data
   for ( unsigned int i = 0 ; i < num_devices ; i ++ ) {
      dev_ptrs[i] = ( DD* ) devices[i].factory( chunk , devices[i].arg );
      chunk += devices[i].dd_size;
   }

   // allocate copy-ins/copy-outs
   CopyData *wdCopies = NULL;
   if ( copies != NULL ) {
      if ( *copies == NULL ) {
         if ( num_copies > 0 ) {
            wdCopies = ( CopyData * ) chunk;
            *copies = wdCopies;
            chunk += num_copies * sizeof( CopyData );
         }
      } else {
         wdCopies = *copies;
      }
   }

   WD * wd =  new (*uwd) WD( num_devices, dev_ptrs, data_size, data != NULL ? *data : NULL, num_copies, num_copies == 0 ? NULL : wdCopies );

   // add to workgroup
   if ( uwg != NULL ) {
      WG * wg = ( WG * )uwg;
      wg->addWork( *wd );
   }

   // set properties
   if ( props != NULL ) {
      if ( props->tied ) wd->tied();
      if ( props->tie_to ) wd->tieTo( *( BaseThread * )props->tie_to );
   }

}

/*! \brief Creates a new Sliced WD
 *
 *  This function creates a new Sliced WD, allocating memory space for device ptrs and
 *  data when necessary. Also allocates Slicer Data object which is related with the WD.
 *
 *  \param [in,out] uwd is the related addr for WD if this parameter is null the
 *                  system will allocate space in memory for the new WD
 *  \param [in] num_devices is the number of related devices
 *  \param [in] devices is a vector of device descriptors 
 *  \param [in] outline_data_size is the size of the related data
 *  \param [in,out] outline_data is the related data (allocated if needed)
 *  \param [in] uwg work group to relate with
 *  \param [in] slicer is the related slicer which contains all the methods to manage
 *              this WD
 *  \param [in] slicer_data_size is the size of the related slicer data
 *  \param [in,out] data used as the slicer data (allocated if needed)
 *  \param [in] props new WD properties
 *
 *  When it does a full allocation the layout is the following:
 *
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
 *  |    copyN      |
 *  +---------------+
 *  |  SlicerData   |
 *  +---------------+
 *
 */
void System::createSlicedWD ( WD **uwd, size_t num_devices, nanos_device_t *devices, size_t outline_data_size,
                        void **outline_data, WG *uwg, Slicer *slicer, size_t slicer_data_size,
                        SlicerData *&slicer_data, nanos_wd_props_t *props, size_t num_copies, nanos_copy_data_t **copies )
{

   int dd_size = 0;
   for ( unsigned int i = 0; i < num_devices; i++ )
      dd_size += devices[i].dd_size;

   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures (temporary solved)
   int size_to_allocate = ( ( *uwd == NULL ) ? sizeof( SlicedWD ) : 0 ) +
                          ( ( outline_data != NULL && *outline_data == NULL ) ? (((outline_data_size+7)>>3)<<3) : 0 ) +
                          ( ( slicer_data == NULL ) ? (((slicer_data_size+7)>>3)<<3) : 0 ) +
                          sizeof( DD* ) * num_devices +
                          dd_size +
                          ( ( copies != NULL && *copies == NULL ) ? num_copies * sizeof(CopyData) : 0 )
                          ;

   char *chunk = 0;

   if ( size_to_allocate ) chunk = new char[size_to_allocate];

   // allocate WD
   if ( *uwd == NULL ) {
      *uwd = ( SlicedWD * ) chunk;
      chunk += sizeof( SlicedWD );
   }

   // allocate WD data
   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures (temporary solved)
   if ( outline_data != NULL && *outline_data == NULL ) {
      *outline_data = chunk;
      chunk += (((outline_data_size+7)>>3)<<3);
   }

   // allocate device pointers vector
   DD **dev_ptrs = ( DD ** ) chunk;
   chunk += sizeof( DD* ) * num_devices;

   // allocate device data
   for ( unsigned int i = 0 ; i < num_devices ; i ++ ) {
      dev_ptrs[i] = ( DD* ) devices[i].factory( chunk , devices[i].arg );
      chunk += devices[i].dd_size;
   }

   // allocate copy-ins/copy-outs
   CopyData *wdCopies = NULL;
   if ( copies != NULL ) {
      if ( *copies == NULL ) {
         if ( num_copies > 0 ) {
            wdCopies = ( CopyData * ) chunk;
            *copies = wdCopies;
            chunk += num_copies * sizeof( CopyData );
         }
      } else {
         wdCopies = *copies;
      }
   }

   // allocate SlicerData
   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures (temporary solved)
   if ( slicer_data == NULL ) {
      slicer_data = ( SlicerData * )chunk;
      chunk += (((slicer_data_size+7)>>3)<<3);
   }

   SlicedWD * wd =  new (*uwd) SlicedWD( *slicer, slicer_data_size, *slicer_data, num_devices, dev_ptrs, 
                       outline_data_size, outline_data != NULL ? *outline_data : NULL, num_copies, num_copies == 0 ? NULL : wdCopies );

   // add to workgroup
   if ( uwg != NULL ) {
      WG * wg = ( WG * )uwg;
      wg->addWork( *wd );
   }

   // set properties
   if ( props != NULL ) {
      if ( props->tied ) wd->tied();
      if ( props->tie_to ) wd->tieTo( *( BaseThread * )props->tie_to );
   }
}

/*! \brief Duplicates a given WD
 *
 *  This function duplicates the given as a parameter WD copying all the
 *  related data (devices ptr, data and DD)
 *
 *  \param [out] uwd is the target addr for the new WD
 *  \param [in] wd is the former WD
 */
void System::duplicateWD ( WD **uwd, WD *wd)
{
   int dd_size = 0;
   void *data = NULL;

   // computing size of device(s)
   for ( unsigned int i = 0; i < wd->getNumDevices(); i++ )
      dd_size += wd->getDevices()[i]->size();

   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures (temporary solved)
   int size_to_allocate = ( ( *uwd == NULL ) ? sizeof( WD ) : 0 ) + (((wd->getDataSize()+7)>>3)<<3) +
                          sizeof( DD* ) * wd->getNumDevices() + dd_size +
                          sizeof( CopyData )* wd->getNumCopies() ;

   char *chunk = 0;

   if ( size_to_allocate ) chunk = new char[size_to_allocate];

   // allocate WD
   if ( *uwd == NULL ) {
      *uwd = ( WD * ) chunk;
      chunk += sizeof( WD );
   }

   // allocate WD data
   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures (temporary solved)
   if ( wd->getDataSize() != 0 ) {
      data = (void * ) chunk;
      memcpy ( data, wd->getData(), wd->getDataSize());
      chunk += (((wd->getDataSize()+7)>>3)<<3);
   }

   // allocate device pointers vector
   DD **dev_ptrs = ( DD ** ) chunk;
   chunk += sizeof( DD* ) * wd->getNumDevices();

   // allocate device data
   for ( unsigned int i = 0 ; i < wd->getNumDevices(); i ++ ) {
      wd->getDevices()[i]->copyTo(chunk);
      dev_ptrs[i] = ( DD * ) chunk;
      chunk += wd->getDevices()[i]->size();
   }

   // allocate copy-in/copy-outs
   CopyData *wdCopies = ( CopyData * ) chunk;
   for ( unsigned int i = 0; i < wd->getNumCopies(); i++ ) {
      CopyData *wdCopiesCurr = ( CopyData * ) chunk;
      *wdCopiesCurr = wd->getCopies()[i];
      chunk += sizeof( CopyData );
   }

   // creating new WD 
   new (*uwd) WD( *wd, dev_ptrs, wdCopies , data);
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
   int dd_size = 0;
   void *data = NULL;
   void *slicer_data = NULL;

   // computing size of device(s)
   for ( unsigned int i = 0; i < wd->getNumDevices(); i++ )
      dd_size += wd->getDevices()[i]->size();

   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures (temporary solved)
   int size_to_allocate = ( ( *uwd == NULL ) ? sizeof( SlicedWD ) : 0 ) + (((wd->getDataSize()+7)>>3)<<3) +
                          sizeof( DD* ) * wd->getNumDevices() + dd_size + (((wd->getSlicerDataSize()+7)>>3)<<3) +
                          sizeof( CopyData )* wd->getNumCopies() ;

   char *chunk = 0;

   if ( size_to_allocate ) chunk = new char[size_to_allocate];

   // allocate WD
   if ( *uwd == NULL ) {
      *uwd = ( SlicedWD * ) chunk;
      chunk += sizeof( SlicedWD );
   }

   // allocate WD data
   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures (temporary solved)
   if ( wd->getDataSize() != 0 ) {
      data = (void * ) chunk;
      memcpy ( data, wd->getData(), wd->getDataSize());
      chunk += (((wd->getDataSize()+7)>>3)<<3);
   }

   // allocate device pointers vector
   DD **dev_ptrs = ( DD ** ) chunk;
   chunk += sizeof( DD* ) * wd->getNumDevices();

   // allocate device data
   for ( unsigned int i = 0 ; i < wd->getNumDevices(); i ++ ) {
      wd->getDevices()[i]->copyTo(chunk);
      dev_ptrs[i] = ( DD * ) chunk;
      chunk += wd->getDevices()[i]->size();
   }

   // allocate copy-in/copy-outs
   CopyData *wdCopies = ( CopyData * ) chunk;
   for ( unsigned int i = 0; i < wd->getNumCopies(); i++ ) {
      CopyData *wdCopiesCurr = ( CopyData * ) chunk;
      *wdCopiesCurr = wd->getCopies()[i];
      chunk += sizeof( CopyData );
   }

   // copy SlicerData
   if ( wd->getSlicerDataSize() != 0 ) {
      slicer_data = (void * ) chunk;
      memcpy ( slicer_data, wd->getSlicerData(), wd->getSlicerDataSize());
      chunk += (((wd->getSlicerDataSize()+7)>>3)<<3);
   }

   // creating new SlicedWD 
   new (*uwd) SlicedWD( *(wd->getSlicer()), wd->getSlicerDataSize(), *((SlicerData *)slicer_data),
                        *((WD *)wd), dev_ptrs, wdCopies, data );

}

void System::setupWD ( WD &work, WD *parent )
{
   work.setParent ( parent );
   work.setDepth( parent->getDepth() +1 );

   // Prepare private copy structures to use relative addresses
   work.prepareCopies();
}

void System::submit ( WD &work )
{
   setupWD( work, myThread->getCurrentWD() );
   work.submit();
}

/*! \brief Submit WorkDescriptor to its parent's  dependencies domain
 */
void System::submitWithDependencies (WD& work, size_t numDeps, Dependency* deps)
{
   setupWD( work, myThread->getCurrentWD() );
   WD *current = myThread->getCurrentWD();
   current->submitWithDependencies( work, numDeps , deps);
}

/*! \brief Wait on the current WorkDescriptor's domain for some dependenices to be satisfied
 */
void System::waitOn( size_t numDeps, Dependency* deps )
{
   WD* current = myThread->getCurrentWD();
   current->waitOn( numDeps, deps );
}


void System::inlineWork ( WD &work )
{
   setupWD( work, myThread->getCurrentWD() );
   // TODO: choose actual (active) device...
   Scheduler::inlineWork( &work );
}


bool System::throttleTask()
{
   return _throttlePolicy->throttle();
}


BaseThread * System:: getUnassignedWorker ( void )
{
   BaseThread *thread;

   for ( unsigned i  = 0; i < _workers.size(); i++ ) {
      if ( !_workers[i]->hasTeam() ) {
         thread = _workers[i];
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
   //TODO: to free or not to free team data?
   debug("Releasing thread " << thread << " from team " << thread->getTeam() );
   thread->leaveTeam();
}

ThreadTeam * System:: createTeam ( unsigned nthreads, void *constraints,
                                   bool reuseCurrent, TeamData *tdata )
{
   int thId;
   TeamData *data;

   if ( nthreads == 0 ) {
      nthreads = 1;
      nthreads = getNumPEs()*getThsPerPE();
   }
   
   SchedulePolicy *sched = 0;
   if ( !sched ) sched = sys.getDefaultSchedulePolicy();

   ScheduleTeamData *stdata = 0;
   if ( sched->getTeamDataSize() > 0 )
      stdata = sched->createTeamData(NULL);

   // create team
   ThreadTeam * team = new ThreadTeam( nthreads, *sched, stdata, *_defBarrFactory() );

   debug( "Creating team " << team << " of " << nthreads << " threads" );

   // find threads
   if ( reuseCurrent ) {
      nthreads --;

      thId = team->addThread( myThread );

      debug( "adding thread " << myThread << " with id " << toString<int>(thId) << " to " << team );

      
      if (tdata) data = &tdata[thId];
      else data = new TeamData();

      ScheduleThreadData *stdata = 0;
      if ( sched->getThreadDataSize() > 0 )
        stdata = sched->createThreadData(NULL);
      
//       data->parentTeam = myThread->getTeamData();

      data->setId(thId);
      data->setScheduleData(stdata);
      
      myThread->enterTeam( team,  data );

      debug( "added thread " << myThread << " with id " << toString<int>(thId) << " to " << team );
   }

   while ( nthreads > 0 ) {
      BaseThread *thread = getUnassignedWorker();

      if ( !thread ) {
         // alex: TODO: create one?
         break;
      }

      nthreads--;
      thId = team->addThread( thread );
      debug( "adding thread " << thread << " with id " << toString<int>(thId) << " to " << team );

      if (tdata) data = &tdata[thId];
      else data = new TeamData();

      ScheduleThreadData *stdata = 0;
      if ( sched->getThreadDataSize() > 0 )
        stdata = sched->createThreadData(NULL);

      data->setId(thId);
      data->setScheduleData(stdata);
      
      thread->enterTeam( team, data );
      debug( "added thread " << thread << " with id " << toString<int>(thId) << " to " << thread->getTeam() );
   }

   team->init();

   return team;
}

void System::endTeam ( ThreadTeam *team )
{
   if ( team->size() > 1 ) {
     fatal("Trying to end a team with running threads");
   }

//    if ( myThread->getTeamData()->parentTeam )
//    {
//       myThread->restoreTeam(myThread->getTeamData()->parentTeam);
//    }

   
   delete team;
}
