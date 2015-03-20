/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#include "threadmanager_decl.hpp"
#include "atomic_decl.hpp"
#include "system.hpp"
#include "config.hpp"
#include "os.hpp"

#ifdef DLB
#include <DLB_interface.h>
#else
extern "C" {
   void DLB_UpdateResources_max( int max_resources ) __attribute__(( weak ));
   void DLB_UpdateResources( void ) __attribute__(( weak ));
   void DLB_ReturnClaimedCpus( void ) __attribute__(( weak ));
   int DLB_ReleaseCpu ( int cpu ) __attribute__(( weak ));
   int DLB_ReturnClaimedCpu ( int cpu ) __attribute__(( weak ));
   void DLB_ClaimCpus (int cpus) __attribute__(( weak ));
   int DLB_CheckCpuAvailability ( int cpu ) __attribute__(( weak ));
   void DLB_Init ( void ) __attribute__(( weak ));
   void DLB_Finalize ( void ) __attribute__(( weak ));
   int DLB_Is_auto( void ) __attribute__(( weak ));
   void DLB_Update( void ) __attribute__(( weak ));
   void DLB_AcquireCpu( int cpu ) __attribute__(( weak ));
}
#define DLB_SYMBOLS_DEFINED ( \
         DLB_UpdateResources_max && \
         DLB_UpdateResources && \
         DLB_ReleaseCpu && \
         DLB_ReturnClaimedCpus && \
         DLB_ReturnClaimedCpu && \
         DLB_ClaimCpus && \
         DLB_CheckCpuAvailability && \
         DLB_Init && \
         DLB_Finalize && \
         DLB_AcquireCpu && \
         DLB_Update )
#endif

using namespace nanos;

const unsigned int ThreadManagerConf::DEFAULT_SLEEP_NS = 20000;
const unsigned int ThreadManagerConf::DEFAULT_YIELDS = 10;

/**********************************/
/****** Thread Manager Conf *******/
/**********************************/

ThreadManagerConf::ThreadManagerConf()
   : _tm(TM_UNDEFINED), _numYields(DEFAULT_YIELDS), _sleepTime(DEFAULT_SLEEP_NS),
   _useYield(false), _useBlock(false), _useDLB(false),
   _forceTieMaster(false), _warmupThreads(false)
{
}

void ThreadManagerConf::config( Config &cfg )
{
   cfg.setOptionsSection("Thread Manager specific","Thread Manager related options");

   Config::MapVar<ThreadManagerOption>* tm_options = NEW Config::MapVar<ThreadManagerOption>( _tm );
   tm_options->addOption( "none", TM_NONE );
   tm_options->addOption( "nanos", TM_NANOS );
   tm_options->addOption( "dlb", TM_DLB );
   cfg.registerConfigOption ( "thread-manager", tm_options, "Select which Thread Manager will be used" );
   cfg.registerArgOption( "thread-manager", "thread-manager" );

   cfg.registerConfigOption( "enable-block", NEW Config::FlagOption( _useBlock, true ),
         "Thread block on idle and condition waits" );
   cfg.registerArgOption( "enable-block", "enable-block" );

   cfg.registerConfigOption( "enable-sleep", NEW Config::FlagOption( _useSleep, true ),
         "Thread sleep on idle and condition waits" );
   cfg.registerArgOption( "enable-sleep", "enable-sleep" );

   cfg.registerConfigOption( "enable-yield", NEW Config::FlagOption( _useYield, true ),
         "Thread yield on idle and condition waits" );
   cfg.registerArgOption( "enable-yield", "enable-yield" );

   std::ostringstream sleep_sstream;
   sleep_sstream << "Set the amount of time (in nsec) in each sleeping phase (default = " << DEFAULT_SLEEP_NS << ")";
   cfg.registerConfigOption ( "sleep-time", NEW Config::UintVar( _sleepTime ), sleep_sstream.str() );
   cfg.registerArgOption ( "sleep-time", "sleep-time" );

   std::ostringstream yield_sstream;
   yield_sstream << "Set number of yields on idle before blocking (default = " << DEFAULT_YIELDS << ")";
   cfg.registerConfigOption ( "num-yields", NEW Config::UintVar( _numYields ), yield_sstream.str() );
   cfg.registerArgOption ( "num-yields", "yields" );

   cfg.registerConfigOption( "enable-dlb", NEW Config::FlagOption ( _useDLB ),
         "Tune Nanos Runtime to be used with Dynamic Load Balancing library" );
   cfg.registerArgOption( "enable-dlb", "enable-dlb" );

   cfg.registerConfigOption( "force-tie-master", NEW Config::FlagOption ( _forceTieMaster ),
         "Force Master WD (user code) to run on Master Thread" );
   cfg.registerArgOption( "force-tie-master", "force-tie-master" );

   cfg.registerConfigOption( "warmup-threads", NEW Config::FlagOption( _warmupThreads, true ),
         "Force the creation of as many threads as available CPUs at initialization time, then block them immediately if needed" );
   cfg.registerArgOption( "warmup-threads", "warmup-threads" );
}

ThreadManager* ThreadManagerConf::create()
{
   // Choose default if _tm not specified
   if ( _tm == TM_UNDEFINED ) {
      if  ( _useYield || _useBlock || _useSleep || _useDLB) {
         _tm = TM_NANOS;
      } else {
         _tm = TM_NONE;
      }
   }

   // Some safety cheks
   if ( _useBlock && _useSleep ) {
      warning0( "Thread Manager: Flags --enable-block and --enable-sleep are mutually exclusive. Block takes precedence." );
      _useSleep = false;
   }
   if ( _tm == TM_NONE && (_useYield || _useBlock || _useSleep || _useDLB) ) {
      warning0( "Thread Manager: Block, sleep, yield or dlb options are ignored when you explicitly choose --thread-manager=none" );
   }
#ifndef DLB
   if ( _useDLB  || _tm == TM_DLB ) {
      fatal_cond0( !DLB_SYMBOLS_DEFINED,
            "Thread Manager: Some DLB option was enabled but DLB symbols were not found. "
            "Either add DLB support at configure time or link your application against DLB libraries." );
   }
#endif

   if ( _tm == TM_NONE ) {
      return NEW ThreadManager( _warmupThreads );
   } else if ( _tm == TM_NANOS ) {
      if ( _useSleep || (_useDLB && !_useBlock) ) {
         return NEW BusyWaitThreadManager( _numYields, _sleepTime, _useSleep, _useDLB, _warmupThreads );
      } else {
         return NEW BlockingThreadManager( _numYields, _useBlock, _useDLB, _warmupThreads );
      }
   } else if ( _tm == TM_DLB ) {
      return NEW DlbThreadManager( _numYields, _warmupThreads );
   }

   fatal0( "Unknown Thread Manager" );
   return NULL;
}

bool ThreadManagerConf::canUntieMaster() const
{
   // If the user forces it, ignore everything else
   if ( _forceTieMaster ) return false;

   const char *lb_policy = OS::getEnvironmentVariable( "LB_POLICY" );
   if ( !_useDLB || lb_policy == NULL ) return true;
   else {
      std::string dlb_policy( lb_policy );
      // Currently auto_LeWI_mask is the only dlb policy that supports untied master
      return (dlb_policy == "auto_LeWI_mask");
   }
}

/**********************************/
/********* Thread Manager *********/
/**********************************/

ThreadManager::ThreadManager( bool warmup )
   : _lock(), _initialized(false), _warmupThreads(warmup)
{}

void ThreadManager::init()
{
   if ( _warmupThreads ) {
      sys.getSMPPlugin()->forceMaxThreadCreation();
   }
   _initialized = true;
}

bool ThreadManager::lastActiveThread()
{
   // We omit the test if the Thread Manager is not yet initializated
   if ( !_initialized ) return false;

   // We omit the test if the cpu does not belong to my process_mask
   BaseThread *thread = getMyThreadSafe();
   int my_cpu = thread->getCpuId();
   if ( !CPU_ISSET( my_cpu, &(sys.getCpuProcessMask()) ) ) return false;

   LockBlock Lock( _lock );
   cpu_set_t mine_and_active;
   CPU_AND( &mine_and_active, &(sys.getCpuProcessMask()), &(sys.getCpuActiveMask()) );

   bool last = CPU_COUNT( &mine_and_active ) == 1 && CPU_ISSET( my_cpu, &mine_and_active );

   if ( last ) {
      // If we get here, my_cpu is the last active, but we must support thread oversubscription
      if ( thread->runningOn()->getRunningThreads() > 1 ) {
         last = false;
      }
   }
   return last;
}

/**********************************/
/**** Blocking Thread Manager *****/
/**********************************/

BlockingThreadManager::BlockingThreadManager( unsigned int num_yields, bool use_block, bool use_dlb, bool warmup )
   : ThreadManager(warmup), _maxCPUs(OS::getMaxProcessors()), _isMalleable(false),
   _numYields(num_yields), _useBlock(use_block), _useDLB(use_dlb)
{}

BlockingThreadManager::~BlockingThreadManager()
{
   if ( _useDLB ) DLB_Finalize();
}

void BlockingThreadManager::init()
{
   ThreadManager::init();
   _isMalleable = sys.getPMInterface().isMalleable();
   if ( _useDLB ) DLB_Init();
}

void BlockingThreadManager::idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
   , unsigned long long& total_yields, unsigned long long& total_blocks
   , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
   )
{
   if ( !_initialized ) return;

   if ( _isMalleable ) {
      acquireResourcesIfNeeded();
   }

   BaseThread *thread = getMyThreadSafe();

   if ( yields > 0 ) {
      NANOS_INSTRUMENT ( total_yields++; )
      NANOS_INSTRUMENT ( unsigned long long begin_yield = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
      thread->yield();
      NANOS_INSTRUMENT ( unsigned long long end_yield = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
      NANOS_INSTRUMENT ( time_yields += ( end_yield - begin_yield ); )
      if ( _useBlock ) yields--;
   } else {
      if ( _useBlock && thread->canBlock() ) {
         NANOS_INSTRUMENT ( total_blocks++; )
         NANOS_INSTRUMENT ( unsigned long long begin_block = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
         releaseCpu();
         NANOS_INSTRUMENT ( unsigned long long end_block = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
         NANOS_INSTRUMENT ( time_blocks += ( end_block - begin_block ); )
         if ( _numYields != 0 ) yields = _numYields;
      }
   }
}

void BlockingThreadManager::acquireResourcesIfNeeded()
{
   if ( !_initialized ) return;

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t ready_tasks_key = ID->getEventKey("concurrent-tasks"); )

   ThreadTeam *team = getMyThreadSafe()->getTeam();

   if ( !team ) return;

   if ( _useDLB ) DLB_Update();

   if ( _isMalleable ) {
      /* OmpSs*/
      int ready_tasks = team->getSchedulePolicy().getPotentiallyParallelWDs();
      if ( ready_tasks > 0 ){

         NANOS_INSTRUMENT( nanos_event_value_t ready_tasks_value = (nanos_event_value_t) ready_tasks )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &ready_tasks_key, &ready_tasks_value); )

         LockBlock Lock( _lock );

         const cpu_set_t& process_mask = sys.getCpuProcessMask();
         cpu_set_t new_active_cpus = sys.getCpuActiveMask();
         cpu_set_t mine_and_active;
         CPU_AND( &mine_and_active, &process_mask, &new_active_cpus );
         // Check first that we have some owned CPU not active
         if ( !CPU_EQUAL(&mine_and_active, &process_mask) ) {
            bool dirty = false;
            // Iterate over default cpus not running and wake them up if needed
            for (int i=0; i<_maxCPUs; i++) {
               if ( CPU_ISSET( i, &process_mask) && !CPU_ISSET( i, &new_active_cpus ) ) {
                  CPU_SET( i, &new_active_cpus );
                  dirty = true;
                  if ( --ready_tasks == 0 )
                     break;
               }
            }
            if (dirty) {
               sys.setCpuActiveMask( &new_active_cpus );
            }
         }
      }
   } else {
      /* OpenMP */
      LockBlock Lock( _lock );
      const cpu_set_t& process_mask = sys.getCpuProcessMask();
      const cpu_set_t& active_mask = sys.getCpuActiveMask();
      if ( !CPU_EQUAL( &process_mask, &active_mask ) ) {
         sys.setCpuActiveMask( &process_mask );
      }
   }
}

void BlockingThreadManager::releaseCpu()
{
   if ( !_initialized ) return;
   if ( !_isMalleable ) return;
   if ( !_useBlock ) return;

   BaseThread *thread = getMyThreadSafe();
   if ( !thread->getTeam() ) return;
   if ( thread->isSleeping() ) return;
   if ( thread->isMainThread() && !sys.getUntieMaster() ) return; /* Do not release master thread if master WD is tied*/

   int my_cpu = thread->getCpuId();

   LockBlock Lock( _lock );

   // Do not release if this CPU is the last active within the process_mask
   cpu_set_t mine_and_active;
   CPU_AND( &mine_and_active, &(sys.getCpuProcessMask()), &(sys.getCpuActiveMask()) );
   if ( CPU_COUNT( &mine_and_active ) == 1 && CPU_ISSET( my_cpu, &mine_and_active ) )
      return;

   // Clear CPU from active mask
   cpu_set_t new_active_cpus = sys.getCpuActiveMask();
   ensure( CPU_ISSET(my_cpu, &new_active_cpus), "Trying to release a non active CPU" );
   CPU_CLR( my_cpu, &new_active_cpus );
   sys.setCpuActiveMask( &new_active_cpus );
}

void BlockingThreadManager::returnClaimedCpus() {}

void BlockingThreadManager::returnMyCpuIfClaimed() {}

void BlockingThreadManager::waitForCpuAvailability() {}

void BlockingThreadManager::acquireThread(BaseThread* thread){} 

/**********************************/
/**** BusyWait Thread Manager *****/
/**********************************/

BusyWaitThreadManager::BusyWaitThreadManager( unsigned int num_yields, unsigned int sleep_time,
                                                bool use_sleep, bool use_dlb, bool warmup )
   : ThreadManager(warmup), _waitingCPUs(), _maxCPUs(OS::getMaxProcessors()), _isMalleable(false),
   _numYields(num_yields), _sleepTime(sleep_time), _useSleep(use_sleep), _useDLB(use_dlb)
{
   CPU_ZERO( &_waitingCPUs );
}

BusyWaitThreadManager::~BusyWaitThreadManager()
{
   if ( _useDLB ) DLB_Finalize();
}

void BusyWaitThreadManager::init()
{
   ThreadManager::init();
   _isMalleable = sys.getPMInterface().isMalleable();
   if ( _useDLB ) DLB_Init();
}

void BusyWaitThreadManager::idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
   , unsigned long long& total_yields, unsigned long long& total_blocks
   , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
   )
{
   if ( !_initialized ) return;

   if ( _useDLB && _isMalleable ) acquireResourcesIfNeeded();

   if ( yields > 0 ) {
      NANOS_INSTRUMENT ( total_yields++; )
      NANOS_INSTRUMENT ( unsigned long long begin_yield = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
      getMyThreadSafe()->yield();
      NANOS_INSTRUMENT ( unsigned long long end_yield = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
      NANOS_INSTRUMENT ( time_yields += ( end_yield - begin_yield ); )
      if ( _useSleep ) yields--;
   } else {
      if ( _useSleep ) {
         NANOS_INSTRUMENT ( total_blocks++; )
         NANOS_INSTRUMENT ( unsigned long begin_block = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
         OS::nanosleep( _sleepTime );
         NANOS_INSTRUMENT ( unsigned long end_block = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
         NANOS_INSTRUMENT ( time_blocks += ( end_block - begin_block ); )
         if ( _numYields != 0 ) yields = _numYields;
      }
   }
}

void BusyWaitThreadManager::acquireResourcesIfNeeded ()
{
   if ( !_initialized ) return;
   if ( !_useDLB ) return;

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t ready_tasks_key = ID->getEventKey("concurrent-tasks"); )

   BaseThread *thread = getMyThreadSafe();
   ThreadTeam *team = thread->getTeam();

   if ( !thread->isMainThread() ) return;
   if ( !team ) return;

   DLB_Update();

   if ( _isMalleable ) {
      /* OmpSs*/
      int ready_tasks = team->getSchedulePolicy().getPotentiallyParallelWDs();
      if ( ready_tasks > 0 ){

         NANOS_INSTRUMENT( nanos_event_value_t ready_tasks_value = (nanos_event_value_t) ready_tasks )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &ready_tasks_key, &ready_tasks_value); )

         LockBlock Lock( _lock );

         int needed_resources = ready_tasks - team->getFinalSize();
         if ( needed_resources > 0 ) {
            // If ready tasks > num threads I claim my cpus being used by someone else
            const cpu_set_t& process_mask = sys.getCpuProcessMask();
            const cpu_set_t& active_mask = sys.getCpuActiveMask();
            cpu_set_t mine_and_active;
            CPU_AND( &mine_and_active, &process_mask, &active_mask );
            if ( !CPU_EQUAL(&mine_and_active, &process_mask) ) {
               // Only claim if some of my CPUs are not active
               DLB_ClaimCpus( needed_resources );
            }

            // If ready tasks > num threads I check if there are available cpus
            needed_resources = ready_tasks - team->getFinalSize();
            if ( needed_resources > 0 ){
               DLB_UpdateResources_max( needed_resources );
            }
         }
      }

   } else {
      /* OpenMP */
      LockBlock Lock( _lock );
      DLB_UpdateResources();
   }
}

void BusyWaitThreadManager::releaseCpu() {}

void BusyWaitThreadManager::returnClaimedCpus()
{
   if ( !_initialized ) return;
   if ( !_useDLB ) return;
   if ( !_isMalleable ) return;
   if ( !getMyThreadSafe()->isMainThread() ) return;

   LockBlock Lock( _lock );

   const cpu_set_t& process_mask = sys.getCpuProcessMask();
   const cpu_set_t& active_mask = sys.getCpuActiveMask();
   cpu_set_t mine_or_active;
   CPU_OR( &mine_or_active, &process_mask, &active_mask );
   if ( CPU_COUNT(&mine_or_active) > CPU_COUNT(&process_mask) ) {
      // Only return if I am using external CPUs
      DLB_ReturnClaimedCpus();
   }
}

void BusyWaitThreadManager::returnMyCpuIfClaimed() {}

void BusyWaitThreadManager::waitForCpuAvailability()
{
   if ( !_initialized ) return;
   if ( !_useDLB ) return;
   int cpu = getMyThreadSafe()->getCpuId();
   CPU_SET( cpu, &_waitingCPUs );
   while ( !lastActiveThread() && !DLB_CheckCpuAvailability(cpu) ) {
      // Sleep and Yield the thread to reduce cycle consumption
      OS::nanosleep( ThreadManagerConf::DEFAULT_SLEEP_NS );
      sched_yield();
   }
   CPU_CLR( cpu, &_waitingCPUs );
}

void BusyWaitThreadManager::acquireThread(BaseThread* thread){} 


/**********************************/
/******* DLB Thread Manager *******/
/**********************************/

DlbThreadManager::DlbThreadManager( unsigned int num_yields, bool warmup )
   : ThreadManager(warmup), _waitingCPUs(), _maxCPUs(OS::getMaxProcessors()),
   _isMalleable(false), _numYields(num_yields)
{
   CPU_ZERO( &_waitingCPUs );
}

DlbThreadManager::~DlbThreadManager()
{
   DLB_Finalize();
}

void DlbThreadManager::init()
{
   ThreadManager::init();
   _isMalleable = sys.getPMInterface().isMalleable();
   DLB_Init();
}

void DlbThreadManager::idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
   , unsigned long long& total_yields, unsigned long long& total_blocks
   , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
   )
{
   if ( !_initialized ) return;

   if ( _isMalleable ) {
      acquireResourcesIfNeeded();
   }

   BaseThread *thread = getMyThreadSafe();

   if ( yields > 0 ) {
      NANOS_INSTRUMENT ( total_yields++; )
      NANOS_INSTRUMENT ( unsigned long long begin_yield = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
      thread->yield();
      NANOS_INSTRUMENT ( unsigned long long end_yield = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
      NANOS_INSTRUMENT ( time_yields += ( end_yield - begin_yield ); )
      yields--;
   } else {
      if ( thread->canBlock() ) {
         // releaseCpu only gives the sleep order, we can skip instrumentation
         //NANOS_INSTRUMENT ( total_blocks++; )
         //NANOS_INSTRUMENT ( unsigned long begin_block = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
         releaseCpu();
         //NANOS_INSTRUMENT ( unsigned long end_block = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
         //NANOS_INSTRUMENT ( time_blocks += ( end_block - begin_block ); )
         if ( _numYields != 0 ) yields = _numYields;
      }
   }
}

void DlbThreadManager::acquireResourcesIfNeeded ()
{
   if ( !_initialized ) return;

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t ready_tasks_key = ID->getEventKey("concurrent-tasks"); )

   ThreadTeam *team = getMyThreadSafe()->getTeam();

   if ( !team ) return;

   DLB_Update();

   if ( _isMalleable ) {
      /* OmpSs*/
      int ready_tasks = team->getSchedulePolicy().getPotentiallyParallelWDs();
      if ( ready_tasks > 0 ){

         NANOS_INSTRUMENT( nanos_event_value_t ready_tasks_value = (nanos_event_value_t) ready_tasks )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &ready_tasks_key, &ready_tasks_value); )

         LockBlock Lock( _lock );

         int needed_resources = ready_tasks - team->getFinalSize();
         if ( needed_resources > 0 ) {
            // If ready tasks > num threads I claim my cpus being used by someone else
            const cpu_set_t& process_mask = sys.getCpuProcessMask();
            const cpu_set_t& active_mask = sys.getCpuActiveMask();
            cpu_set_t mine_and_active;
            CPU_AND( &mine_and_active, &process_mask, &active_mask );
            if ( !CPU_EQUAL(&mine_and_active, &process_mask) ) {
               // Only claim if some of my CPUs are not active
               DLB_ClaimCpus( needed_resources );
            }

            // If ready tasks > num threads I check if there are available cpus
            needed_resources = ready_tasks - team->getFinalSize();
            if ( needed_resources > 0 ){
               DLB_UpdateResources_max( needed_resources );
            }
         }
      }

   } else {
      /* OpenMP */
      LockBlock Lock( _lock );
      DLB_UpdateResources();
   }
}

void DlbThreadManager::releaseCpu()
{
   if ( !_initialized ) return;
   if ( !_isMalleable ) return;

   BaseThread *thread = getMyThreadSafe();
   if ( !thread->getTeam() ) return;
   if ( thread->isSleeping() ) return;

   if ( thread->isMainThread() && !sys.getUntieMaster() ) return; /* Do not release master thread if master WD is tied*/

   int my_cpu = thread->getCpuId();

   LockBlock Lock( _lock );

   // Do not release if this CPU is the last active within the process_mask
   cpu_set_t mine_and_active;
   CPU_AND( &mine_and_active, &(sys.getCpuProcessMask()), &(sys.getCpuActiveMask()) );
   if ( CPU_COUNT( &mine_and_active ) == 1 && CPU_ISSET( my_cpu, &mine_and_active ) )
      return;

   DLB_ReleaseCpu( my_cpu );
}

void DlbThreadManager::returnClaimedCpus() {}

void DlbThreadManager::returnMyCpuIfClaimed()
{
   if ( !_initialized ) return;
   if ( !_isMalleable ) return;

   // Return if my cpu belongs to the default mask
   BaseThread *thread = getMyThreadSafe();
   const cpu_set_t& process_mask = sys.getCpuProcessMask();
   int my_cpu = thread->getCpuId();
   if ( CPU_ISSET( my_cpu, &process_mask ) )
      return;

   if ( !thread->isSleeping() ) {
      LockBlock Lock( _lock );
      DLB_ReturnClaimedCpu( my_cpu );
   }
}

void DlbThreadManager::waitForCpuAvailability()
{
   int cpu = getMyThreadSafe()->getCpuId();
   CPU_SET( cpu, &_waitingCPUs );
   while ( !lastActiveThread() && !DLB_CheckCpuAvailability(cpu) ) {
      // Sleep and Yield the thread to reduce cycle consumption
      OS::nanosleep( ThreadManagerConf::DEFAULT_SLEEP_NS );
      sched_yield();
   }
   CPU_CLR( cpu, &_waitingCPUs );
}

void DlbThreadManager::acquireThread(BaseThread* thread) {
   int cpu= thread->getCpuId();
   if (!CPU_ISSET(cpu, &(sys.getCpuActiveMask()))){
    //If the cpu is not active claim it and wake up thread
        DLB_AcquireCpu( cpu );        
    }
}
