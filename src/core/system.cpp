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

using namespace nanos;

System nanos::sys;

// default system values go here
System::System () : _numPEs( 1 ), _deviceStackSize( 1024 ), _bindThreads( true ), _profile( false ), _instrument( false ),
      _verboseMode( false ), _executionMode( DEDICATED ), _thsPerPE( 1 ),
      _defSchedule( "cilk" ), _defThrottlePolicy( "numtasks" ), _defBarr( "posix" )
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

   verbose0( "loading defult cutoff policy" );

   if ( !PluginManager::load( "throttle-"+getDefaultThrottlePolicy() ) )
      fatal0( "Could not load main cutoff policy" );

   verbose0( "loading default barrier algorithm" );

   if ( !PluginManager::load( "barrier-"+getDefaultBarrier() ) )
      fatal0( "Could not load main barrier algorithm" );

   ensure( _defBarrFactory,"No default system barrier factory" );

}


void System::config ()
{
   Config config;

   verbose0 ( "Preparing configuration" );
   config.registerArgOption( new Config::PositiveVar( "nth-pes",_numPEs ) );
   config.registerEnvOption( new Config::PositiveVar( "NTH_PES",_numPEs ) );
   config.registerArgOption( new Config::PositiveVar( "nth-stack-size",_deviceStackSize ) );
   config.registerEnvOption( new Config::PositiveVar( "NTH_STACK_SIZE",_deviceStackSize ) );
   config.registerArgOption( new Config::FlagOption( "nth-no-binding", _bindThreads, false ) );
   config.registerArgOption( new Config::FlagOption( "nth-verbose",_verboseMode ) );

   //more than 1 thread per pe
   config.registerArgOption( new Config::PositiveVar( "nth-thsperpe",_thsPerPE ) );

   //TODO: how to simplify this a bit?
   Config::MapVar<ExecutionMode>::MapList opts( 2 );
   opts[0] = Config::MapVar<ExecutionMode>::MapOption( "dedicated",DEDICATED );
   opts[1] = Config::MapVar<ExecutionMode>::MapOption( "shared",SHARED );
   config.registerArgOption(
      new Config::MapVar<ExecutionMode>( "nth-mode",_executionMode,opts ) );

   config.registerArgOption ( new Config::StringVar ( "nth-schedule", _defSchedule ) );
   config.registerEnvOption ( new Config::StringVar ( "NTH_SCHEDULE", _defSchedule ) );

   config.registerArgOption ( new Config::StringVar ( "nth-throttle", _defThrottlePolicy ) );
   config.registerEnvOption ( new Config::StringVar ( "NTH_TROTTLE", _defThrottlePolicy ) );

   config.registerArgOption ( new Config::StringVar ( "nth-barrier", _defBarr ) );
   config.registerEnvOption ( new Config::StringVar ( "NTH_BARRIER", _defBarr ) );

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
   _workers.push_back( &pe->associateThisThread ( sg ) );


   //starting as much threads per pe as requested by the user

   for ( int ths = 1; ths < getThsPerPE(); ths++ ) {
      pe->startWorker( sg );
   }

   for ( int p = 1; p < numPes ; p++ ) {
      pe = createPE ( "smp", p );
      _pes.push_back ( pe );

      //starting as much threads per pe as requested by the user

      for ( int ths = 0; ths < getThsPerPE(); ths++ ) {
         _workers.push_back( &pe->startWorker( sg ) );
      }
   }

   // count one for the "main" task
   sys._numTasksRunning=1;

   createTeam( numPes*getThsPerPE() );
}

System::~System ()
{
   return;
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

   for ( unsigned p = 1; p < _pes.size() ; p++ ) {
      delete _pes[p];
   }

   verbose ( "NANOS++ shutting down.... end" );
}

void System::submit ( WD &work )
{
   work.setParent ( myThread->getCurrentWD() );
   work.setDepth( work.getParent()->getDepth() +1 );
   Scheduler::submit ( work );
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
   thread->leaveTeam();
}

ThreadTeam * System:: createTeam ( int nthreads, SG *policy, void *constraints, bool reuseCurrent )
{
   int thId = 0;
   if ( !policy ) policy = _defSGFactory( nthreads );

   // create team
   ThreadTeam * team = new ThreadTeam( nthreads, *policy, *_defBarrFactory() );

   debug( "Creating team " << team << " of " << nthreads << " threads" );

   // find threads
   if ( reuseCurrent ) {
      debug( "adding thread " << myThread << " with id " << toString<int>(thId) << " to " << team );
      
      nthreads --;
      team->addThread( myThread );
      myThread->enterTeam( team, thId++ );
   }

   while ( nthreads > 0 ) {
      BaseThread *thread = getUnassignedWorker();

      if ( !thread ) {
         // TODO: create one?
         break;
      }

      debug( "adding thread " << thread << " with id " << toString<int>(thId) << " to " << team );

      nthreads--;
      team->addThread( thread );
      thread->enterTeam( team, thId++ );
   }

   team->init();

   return team;
}

