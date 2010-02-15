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

#ifdef SPU_DEV
#include "spuprocessor.hpp"
#endif

using namespace nanos;

System nanos::sys;

// default system values go here
System::System () : _numPEs( 1 ), _deviceStackSize( 1024 ), _bindThreads( true ), _profile( false ), _instrument( false ),
      _verboseMode( false ), _executionMode( DEDICATED ), _thsPerPE( 1 ), _untieMaster(true),
      _defSchedule( "bf" ), _defThrottlePolicy( "numtasks" ), _defBarr( "posix" ), _defInstr ( "empty_trace" ),
      _instrumentor ( NULL )
{
   verbose0 ( "NANOS++ initalizing... start" );
   config();
   loadModules();
   start();
   verbose0 ( "NANOS++ initalizing... end" );
}

void System::loadModules ()
{
   verbose0 ( "Configuring module manager" );

   PluginManager::init();

   verbose0 ( "Loading modules" );

   // load host processor module
   verbose0( "loading SMP support" );

   if ( !PluginManager::load ( "pe-smp" ) )
      fatal0 ( "Couldn't load SMP support" );

   ensure( _hostFactory,"No default smp factory" );

   // load default schedule plugin
   verbose0( "loading " << getDefaultSchedule() << " scheduling policy support" );

   if ( !PluginManager::load ( "sched-"+getDefaultSchedule() ) )
      fatal0 ( "Couldn't load main scheduling policy" );

   ensure( _defSGFactory,"No default system scheduling factory" );

   verbose0( "loading " << getDefaultThrottlePolicy() << " throttle policy" );

   if ( !PluginManager::load( "throttle-"+getDefaultThrottlePolicy() ) )
      fatal0( "Could not load main cutoff policy" );

   ensure(_throttlePolicy, "No default throttle policy");

   verbose0( "loading " << getDefaultBarrier() << " barrier algorithm" );

   if ( !PluginManager::load( "barrier-"+getDefaultBarrier() ) )
      fatal0( "Could not load main barrier algorithm" );

   if ( !PluginManager::load( "instrumentor-"+getDefaultInstrumentor() ) )
      fatal0( "Could not load " + getDefaultInstrumentor() + " instrumentor" );


   ensure( _defBarrFactory,"No default system barrier factory" );

}


void System::config ()
{
   Config config;

   verbose0 ( "Preparing configuration" );

   config.setOptionsSection ( "Global options", new std::string( "Global options for the Nanox runtime" ) );

   config.registerConfigOption ( "num_pes", new Config::PositiveVar( _numPEs ), "Number of processing elements" );
   config.registerArgOption ( "num_pes", "pes" );
   config.registerEnvOption ( "num_pes", "NX_PES" );

   config.registerConfigOption ( "stack-size", new Config::PositiveVar( _deviceStackSize ), "Default stack size for the devices" );
   config.registerArgOption ( "stack-size", "stack-size" );
   config.registerEnvOption ( "stack-size", "NX_STACK_SIZE" );

   config.registerConfigOption ( "no-binding", new Config::FlagOption( _bindThreads, false), "Thread binding" );
   config.registerArgOption ( "no-binding", "no-binding" );

   config.registerConfigOption ( "verbose", new Config::FlagOption( _verboseMode), "Verbose mode" );
   config.registerArgOption ( "verbose", "verbose" );

   //more than 1 thread per pe
   config.registerConfigOption ( "thsperpe", new Config::PositiveVar( _thsPerPE ), "Number of threads per processing element" );
   config.registerArgOption ( "thsperpe", "thrsperpe" );

   Config::MapVar<ExecutionMode> map( _executionMode );
   map.addOption( "dedicated", DEDICATED).addOption( "shared", SHARED );
   config.registerConfigOption ( "exec_mode", &map, "Execution mode" );
   config.registerArgOption ( "exec_mode", "mode" );

   config.registerConfigOption ( "schedule", new Config::StringVar ( _defSchedule ), "Default scheduling policy" );
   config.registerArgOption ( "schedule", "schedule" );
   config.registerEnvOption ( "schedule", "NX_SCHEDULE" );

   config.registerConfigOption ( "throttle", new Config::StringVar ( _defThrottlePolicy ), "Default throttle policy" );
   config.registerArgOption ( "throttle", "throttle" );
   config.registerEnvOption ( "throttle", "NX_THROTTLE" );

   config.registerConfigOption ( "barrier", new Config::StringVar ( _defBarr ), "Default barrier" );
   config.registerArgOption ( "barrier", "barrier" );
   config.registerEnvOption ( "barrier", "NX_BARRIER" );

   config.registerConfigOption ( "instrumentor", new Config::StringVar ( _defInstr ), "Nanos instrumentation" );
   config.registerArgOption ( "instrumentor", "instrumentor" );
   config.registerEnvOption ( "instrumentor", "NX_INSTRUMENTOR" );

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
   verbose0 ( "Starting threads" );

   int numPes = getNumPEs();

   _pes.reserve ( numPes );

   SchedulingGroup *sg = _defSGFactory( numPes*getThsPerPE() );

   //TODO: decide, single master, multiple master start
   PE *pe = createPE ( "smp", 0 );
   _pes.push_back ( pe );
   _workers.push_back( &pe->associateThisThread ( sg, _untieMaster ) );

   getInstrumentor()->initialize();

   //start as much threads per pe as requested by the user
   for ( int ths = 1; ths < getThsPerPE(); ths++ ) {
      _workers.push_back( &pe->startWorker( sg ));
   }

   for ( int p = 1; p < numPes ; p++ ) {
      pe = createPE ( "smp", p );
      _pes.push_back ( pe );

      //starting as much threads per pe as requested by the user

      for ( int ths = 0; ths < getThsPerPE(); ths++ ) {
         _workers.push_back( &pe->startWorker( sg ) );
      }
   }
   
#ifdef SPU_DEV
   PE *spu = new nanos::ext::SPUProcessor(100, (nanos::ext::SMPProcessor &) *_pes[0]);
   spu->startWorker(sg);
#endif

   // count one for the "main" task
   sys._numTasksRunning=1;

   if (0) // TODO: mode variable
     createTeam( numPes*getThsPerPE() );
   else
     createTeam(1);
}

System::~System ()
{
   verbose ( "NANOS++ shutting down.... init" );

   verbose ( "Wait for main workgroup to complete" );
   myThread->getCurrentWD()->waitCompletation();

   verbose ( "Joining threads... phase 1" );
   // signal stop PEs

   for ( unsigned p = 1; p < _pes.size() ; p++ ) {
      _pes[p]->stopAll();
   }

   verbose ( "Joining threads... phase 2" );

   // join
   getInstrumentor()->finalize();

   for ( unsigned p = 1; p < _pes.size() ; p++ ) {
      delete _pes[p];
   }

   verbose ( "NANOS++ shutting down.... end" );
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
 *
 */
void System::createWD ( WD **uwd, size_t num_devices, nanos_device_t *devices, size_t data_size,
                        void **data, WG *uwg, nanos_wd_props_t *props )
{
   int dd_size = 0;
   for ( unsigned int i = 0; i < num_devices; i++ )
      dd_size += devices[i].dd_size;

   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures (temporary solved)
   int size_to_allocate = ( ( *uwd == NULL ) ? sizeof( WD ) : 0 ) +
                          ( ( data != NULL && *data == NULL ) ? (((data_size+7)>>3)<<3) : 0 ) +
                          sizeof( DD* ) * num_devices +
                          dd_size
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

   WD * wd =  new (*uwd) WD( num_devices, dev_ptrs, data_size, data != NULL ? *data : NULL );

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
 *  |  SlicerData   |
 *  +---------------+
 *
 */
void System::createSlicedWD ( WD **uwd, size_t num_devices, nanos_device_t *devices, size_t outline_data_size,
                        void **outline_data, WG *uwg, Slicer *slicer, size_t slicer_data_size,
                        SlicerData *&slicer_data, nanos_wd_props_t *props )
{

   int dd_size = 0;
   for ( unsigned int i = 0; i < num_devices; i++ )
      dd_size += devices[i].dd_size;

   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures (temporary solved)
   int size_to_allocate = ( ( *uwd == NULL ) ? sizeof( SlicedWD ) : 0 ) +
                          ( ( outline_data != NULL && *outline_data == NULL ) ? (((outline_data_size+7)>>3)<<3) : 0 ) +
                          ( ( slicer_data == NULL ) ? (((slicer_data_size+7)>>3)<<3) : 0 ) +
                          sizeof( DD* ) * num_devices +
                          dd_size
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

   // allocate SlicerData
   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures (temporary solved)
   if ( slicer_data == NULL ) {
      slicer_data = ( SlicerData * )chunk;
      chunk += (((slicer_data_size+7)>>3)<<3);
   }

   SlicedWD * wd =  new (*uwd) SlicedWD( *slicer, slicer_data_size, *slicer_data, num_devices, dev_ptrs, 
                       outline_data_size, outline_data != NULL ? *outline_data : NULL );

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
                          sizeof( DD* ) * wd->getNumDevices() + dd_size ;

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

   // creating new WD 
   new (*uwd) WD( *wd, dev_ptrs, data );
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
                          sizeof( DD* ) * wd->getNumDevices() + dd_size + (((wd->getSlicerDataSize()+7)>>3)<<3);

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

   // copy SlicerData
   if ( wd->getSlicerDataSize() != 0 ) {
      slicer_data = (void * ) chunk;
      memcpy ( slicer_data, wd->getSlicerData(), wd->getSlicerDataSize());
      chunk += (((wd->getSlicerDataSize()+7)>>3)<<3);
   }

   // creating new SlicedWD 
   new (*uwd) SlicedWD( *(wd->getSlicer()), wd->getSlicerDataSize(), *((SlicerData *)slicer_data),
                        *((WD *)wd), dev_ptrs, data );

}

void System::submit ( WD &work )
{
   work.setParent ( myThread->getCurrentWD() );
   work.setDepth( work.getParent()->getDepth() +1 );
   work.submit();
}

/*! \brief Submit WorkDescriptor to its parent's  dependencies domain
 */
void System::submitWithDependencies (WD& work, size_t numDeps, Dependency* deps)
{
   WD *current = myThread->getCurrentWD();
   work.setParent ( current );
   work.setDepth( work.getParent()->getDepth() +1 );
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
   BaseThread *myself = myThread;

   // TODO: choose actual device...
   work.setParent ( myself->getCurrentWD() );
   myself->inlineWork( &work );
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

ThreadTeam * System:: createTeam ( unsigned nthreads, SG *policy, void *constraints,
                                   bool reuseCurrent, TeamData *tdata )
{
   int thId;
   TeamData *data;

   if ( nthreads == 0 ) {
      nthreads = 1;
      nthreads = getNumPEs()*getThsPerPE();
   }
   
   if ( !policy ) policy = _defSGFactory( nthreads );

   // create team
   ThreadTeam * team = new ThreadTeam( nthreads, *policy, *_defBarrFactory() );

   debug( "Creating team " << team << " of " << nthreads << " threads" );

   // find threads
   if ( reuseCurrent ) {
      
      nthreads --;

      thId = team->addThread( myThread );

      debug( "adding thread " << myThread << " with id " << toString<int>(thId) << " to " << team );
      if (tdata) data = &tdata[thId];
      else data = new TeamData();

      data->setId(thId);
      myThread->enterTeam( team,  data );
   }

   while ( nthreads > 0 ) {
      BaseThread *thread = getUnassignedWorker();

      if ( !thread ) {
         // TODO: create one?
         break;
      }

      nthreads--;
      thId = team->addThread( thread );
      debug( "adding thread " << thread << " with id " << toString<int>(thId) << " to " << team );

      if (tdata) data = &tdata[thId];
      else data = new TeamData();

      data->setId(thId);
      thread->enterTeam( team, data );
   }

   team->init();

   return team;
}

void System::endTeam ( ThreadTeam *team )
{
   if ( team->size() > 1 ) {
     fatal("Trying to end a team with running threads");
   }

   
   delete team;
}
