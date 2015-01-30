/*************************************************************************************/
/*      Copyright 2013 Barcelona Supercomputing Center                               */
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
         DLB_Update )

#endif

// Sleep time in ns between each sched_yield
#define NANOS_YIELD_SLEEP_NS 20000


using namespace nanos;

/**********************************/
/****** Thread Manager Conf *******/
/**********************************/

ThreadManagerConf::ThreadManagerConf()
   : _tm(), _numYields(1), _useYield(false), _useBlock(false), _useDLB(false)
{
}

void ThreadManagerConf::config( Config &cfg )
{
   cfg.setOptionsSection("Thread Manager specific","Thread Manager related options");

   Config::MapVar<ThreadManagerOption>* tm_options = NEW Config::MapVar<ThreadManagerOption>( _tm );
   tm_options->addOption( "none", NONE );
   tm_options->addOption( "basic", BASIC );
   tm_options->addOption( "basic-dlb", BASIC_DLB );
   tm_options->addOption( "auto-dlb", AUTO_DLB );
   cfg.registerConfigOption ( "thread-manager", tm_options, "Selects which Thread Manager will be used" );
   cfg.registerArgOption( "thread-manager", "thread-manager" );

   cfg.registerConfigOption( "enable-yield", NEW Config::FlagOption( _useYield, true ),
                             "Thread yield on idle and condition waits (default is disabled)" );
   cfg.registerArgOption( "enable-yield", "enable-yield" );

   cfg.registerConfigOption( "enable-block", NEW Config::FlagOption( _useBlock, true ),
                             "Thread block on idle and condition waits (default is disabled)" );
   cfg.registerArgOption( "enable-block", "enable-block" );

   cfg.registerConfigOption ( "num-yields", NEW Config::UintVar( _numYields ),
         "Set number of yields on Idle before block (default = 1)" );
   cfg.registerArgOption ( "num-yields", "yields" );

   cfg.registerConfigOption( "enable-dlb", NEW Config::FlagOption ( _useDLB ),
                              "Tune Nanos Runtime to be used with Dynamic Load Balancing library)" );
   cfg.registerArgOption( "enable-dlb", "enable-dlb" );
}

ThreadManager* ThreadManagerConf::create()
{
   if ( _tm == UNDEFINED ) {
      if  ( _useYield || _useBlock ) {
         // yield or block enabled implies a basic thread manager
         _tm = BASIC;
      } else {
         // default Thread Manager
         _tm = NONE;
      }
   }

   if ( _tm == NONE && (_useYield || _useBlock || _useDLB) ) {
      warning0( "Block, yield or dlb options are ignored when you explicitly choose --thread-manager=none" );
   }

   if ( _tm == NONE ) {
      return NEW ThreadManager();
   } else if ( _tm == BASIC ) {
      return NEW BasicThreadManager( _numYields, _useYield, _useBlock, _useDLB );
   } else if ( _tm == BASIC_DLB ) {
      return NEW BasicDlbThreadManager( _numYields, _useYield, _useBlock, _useDLB );
   } else if ( _tm == AUTO_DLB ) {
      return NEW AutoDlbThreadManager( _numYields, _useYield, _useBlock, _useDLB );
   }

   fatal0( "Unknown Thread Manager" );
   return NULL;
}

bool ThreadManagerConf::canUntieMaster() const
{
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

/* Non virtual methods, common to all Thread Managers */

bool ThreadManager::lastActiveThread()
{
   // We omit the test if the cpu does not belong to my process_mask
   int my_cpu = getMyThreadSafe()->getCpuId();
   if ( !CPU_ISSET( my_cpu, &(sys.getCpuProcessMask()) ) ) return false;

   LockBlock Lock( _lock );
   cpu_set_t mine_and_active;
   CPU_AND( &mine_and_active, &(sys.getCpuProcessMask()), &(sys.getCpuActiveMask()) );
   return ( CPU_COUNT( &mine_and_active ) == 1 && CPU_ISSET( my_cpu, &mine_and_active ) );
}

/**********************************/
/****** Basic Thread Manager ******/
/**********************************/

BasicThreadManager::BasicThreadManager( unsigned int num_yields, bool use_yield,
                                          bool use_block, bool use_dlb )
   : ThreadManager(), _maxCPUs(OS::getMaxProcessors()),
   _isMalleable(sys.getPMInterface().isMalleable()), _numYields(num_yields),
   _useYield(use_yield), _useBlock(use_block), _useDLB(use_dlb)
{
#ifndef DLB
   if (_useDLB) {
      fatal_cond0( !DLB_SYMBOLS_DEFINED, "Flag --enable-dlb detected but DLB symbols were not found" );
   }
#endif
   if ( _useDLB) DLB_Init();
}

BasicThreadManager::~BasicThreadManager()
{
   if ( _useDLB) DLB_Finalize();
}

void BasicThreadManager::idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
   , unsigned long long& total_yields, unsigned long long& total_blocks
   , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
   )
{
   if ( _isMalleable ) {
      acquireResourcesIfNeeded();
   }

   BaseThread *thread = getMyThreadSafe();

   if ( !_useYield || yields == 0 ) {
      if ( _useBlock && thread->canBlock() ) {
         NANOS_INSTRUMENT ( total_blocks++; )
         NANOS_INSTRUMENT ( unsigned long long begin_block = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
         releaseCpu();
         NANOS_INSTRUMENT ( unsigned long long end_block = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
         NANOS_INSTRUMENT ( time_blocks += ( end_block - begin_block ); )
      }

   } else if ( _useYield ) {
      NANOS_INSTRUMENT ( total_yields++; )
      NANOS_INSTRUMENT ( unsigned long long begin_yield = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
      thread->yield();
      NANOS_INSTRUMENT ( unsigned long long end_yield = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
      NANOS_INSTRUMENT ( time_yields += ( end_yield - begin_yield ); )
      if ( _useBlock ) yields--;
   }
}

void BasicThreadManager::acquireResourcesIfNeeded()
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t ready_tasks_key = ID->getEventKey("concurrent-tasks"); )

   ThreadTeam *team;

   if ( !(team = getMyThreadSafe()->getTeam()) ) return;

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

void BasicThreadManager::releaseCpu()
{
   if ( !_isMalleable ) return;
   if ( !_useBlock ) return;

   BaseThread *thread = getMyThreadSafe();
   if ( !thread->getTeam() ) return;
   if ( thread->isSleeping() ) return;

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

void BasicThreadManager::returnClaimedCpus() {}

void BasicThreadManager::returnMyCpuIfClaimed() {}

void BasicThreadManager::waitForCpuAvailability() {}


/**********************************/
/**** Basic DLB Thread Manager ****/
/**********************************/

BasicDlbThreadManager::BasicDlbThreadManager( unsigned int num_yields, bool use_yield,
                                          bool use_block, bool use_dlb )
   : ThreadManager(), _waitingCPUs(), _maxCPUs(OS::getMaxProcessors()),
   _isMalleable(sys.getPMInterface().isMalleable()), _numYields(num_yields),
   _useYield(use_yield), _useBlock(use_block), _useDLB(true)
{
#ifndef DLB
   fatal_cond0( !DLB_SYMBOLS_DEFINED, "Using the DLB thread manager but DLB symbols were not found" );
#endif
   LockBlock Lock( _lock );
   CPU_ZERO( &_waitingCPUs );

   DLB_Init();
}

BasicDlbThreadManager::~BasicDlbThreadManager()
{
   DLB_Finalize();
}

void BasicDlbThreadManager::idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
   , unsigned long long& total_yields, unsigned long long& total_blocks
   , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
   )
{
   if ( _isMalleable ) acquireResourcesIfNeeded();

   if ( !_useYield || yields == 0 ) {
      if ( _useBlock ) {
         NANOS_INSTRUMENT ( total_blocks++; )
         NANOS_INSTRUMENT ( unsigned long begin_block = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
         // we cannot release the cpu using basic DLB, just get out of the cpu ready queue for a little while
         OS::nanosleep( NANOS_YIELD_SLEEP_NS );
         NANOS_INSTRUMENT ( unsigned long end_block = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
         NANOS_INSTRUMENT ( time_blocks += ( end_block - begin_block ); )
         yields = _numYields;
      }
   } else if ( _useYield ) {
      NANOS_INSTRUMENT ( total_yields++; )
      NANOS_INSTRUMENT ( unsigned long long begin_yield = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
      getMyThreadSafe()->yield();
      NANOS_INSTRUMENT ( unsigned long long end_yield = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
      NANOS_INSTRUMENT ( time_yields += ( end_yield - begin_yield ); )
      if ( _useBlock ) yields--;
   }
}

void BasicDlbThreadManager::acquireResourcesIfNeeded ()
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t ready_tasks_key = ID->getEventKey("concurrent-tasks"); )

   ThreadTeam *team;
   BaseThread *thread = getMyThreadSafe();

   if ( !thread->isMainThread() ) return;

   if ( !(team = thread->getTeam()) ) return;

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

void BasicDlbThreadManager::releaseCpu() {}

void BasicDlbThreadManager::returnClaimedCpus()
{
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

void BasicDlbThreadManager::returnMyCpuIfClaimed() {}

void BasicDlbThreadManager::waitForCpuAvailability()
{
   int cpu = getMyThreadSafe()->getCpuId();
   CPU_SET( cpu, &_waitingCPUs );
   while ( !lastActiveThread() && !DLB_CheckCpuAvailability(cpu) ) {
      // Sleep and Yield the thread to reduce cycle consumption
      OS::nanosleep( NANOS_YIELD_SLEEP_NS );
      sched_yield();
   }
   CPU_CLR( cpu, &_waitingCPUs );
}

/**********************************/
/**** Auto DLB Thread Manager *****/
/**********************************/

AutoDlbThreadManager::AutoDlbThreadManager( unsigned int num_yields, bool use_yield,
                                          bool use_block, bool use_dlb )
   : ThreadManager(), _waitingCPUs(), _maxCPUs(OS::getMaxProcessors()),
   _isMalleable(sys.getPMInterface().isMalleable()), _numYields(num_yields),
   _useYield(use_yield), _useBlock(true), _useDLB(true)
{
#ifndef DLB
   fatal_cond0( !DLB_SYMBOLS_DEFINED, "Using the DLB thread manager but DLB symbols were not found" );
#endif
   LockBlock Lock( _lock );
   CPU_ZERO( &_waitingCPUs );

   DLB_Init();
}

AutoDlbThreadManager::~AutoDlbThreadManager()
{
   DLB_Finalize();
}

void AutoDlbThreadManager::idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
   , unsigned long long& total_yields, unsigned long long& total_blocks
   , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
   )
{
   if ( _isMalleable ) {
      acquireResourcesIfNeeded();
   }

   BaseThread *thread = getMyThreadSafe();

   if ( !_useYield || yields == 0 ) {
      if ( _useBlock && thread->canBlock() ) {
         NANOS_INSTRUMENT ( total_blocks++; )
         NANOS_INSTRUMENT ( unsigned long begin_block = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
         releaseCpu();
         NANOS_INSTRUMENT ( unsigned long end_block = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
         NANOS_INSTRUMENT ( time_blocks += ( end_block - begin_block ); )
      }

   } else if ( _useYield ) {
      NANOS_INSTRUMENT ( total_yields++; )
      NANOS_INSTRUMENT ( unsigned long long begin_yield = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
      thread->yield();
      NANOS_INSTRUMENT ( unsigned long long end_yield = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
      NANOS_INSTRUMENT ( time_yields += ( end_yield - begin_yield ); )
      if ( _useBlock ) yields--;
   }
}

void AutoDlbThreadManager::acquireResourcesIfNeeded ()
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t ready_tasks_key = ID->getEventKey("concurrent-tasks"); )

   ThreadTeam *team;

   if ( !(team = getMyThreadSafe()->getTeam()) ) return;

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

void AutoDlbThreadManager::releaseCpu()
{
   if ( !_isMalleable ) return;

   BaseThread *thread = getMyThreadSafe();
   if ( !thread->getTeam() ) return;
   if ( thread->isSleeping() ) return;

   int my_cpu = thread->getCpuId();

   LockBlock Lock( _lock );

   // Do not release if this CPU is the last active within the process_mask
   cpu_set_t mine_and_active;
   CPU_AND( &mine_and_active, &(sys.getCpuProcessMask()), &(sys.getCpuActiveMask()) );
   if ( CPU_COUNT( &mine_and_active ) == 1 && CPU_ISSET( my_cpu, &mine_and_active ) )
      return;

   DLB_ReleaseCpu( my_cpu );
}

void AutoDlbThreadManager::returnClaimedCpus() {}

void AutoDlbThreadManager::returnMyCpuIfClaimed()
{
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

void AutoDlbThreadManager::waitForCpuAvailability()
{
   int cpu = getMyThreadSafe()->getCpuId();
   CPU_SET( cpu, &_waitingCPUs );
   while ( !lastActiveThread() && !DLB_CheckCpuAvailability(cpu) ) {
      // Sleep and Yield the thread to reduce cycle consumption
      OS::nanosleep( NANOS_YIELD_SLEEP_NS );
      sched_yield();
   }
   CPU_CLR( cpu, &_waitingCPUs );
}
