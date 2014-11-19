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

#include "resourcemanager.hpp"
#include "system.hpp"
#include "atomic_decl.hpp"
#include "basethread_decl.hpp"
#include "threadteam_decl.hpp"
#include "instrumentation_decl.hpp"
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
}
#endif

namespace nanos {
namespace ResourceManager {
   namespace {
      typedef struct flags_t {
         bool initialized:1;
         bool is_malleable:1;
         bool dlb_enabled:1;
         bool block_enabled:1;
      } flags_t;

      flags_t   _status = {false, false, false, false};
      Lock      _lock;
      cpu_set_t _waiting_cpus;
      int       _max_cpus;
   }
}}

using namespace nanos;


void ResourceManager::init( void )
{
   LockBlock Lock( _lock );
   CPU_ZERO( &_waiting_cpus );
   _max_cpus = OS::getMaxProcessors();

   _status.is_malleable = sys.getPMInterface().isMalleable();
   _status.block_enabled = sys.getSchedulerConf().getUseBlock();
   _status.dlb_enabled = sys.dlbEnabled();
#ifndef DLB
   _status.dlb_enabled &=
         DLB_UpdateResources_max &&
         DLB_UpdateResources &&
         DLB_ReleaseCpu &&
         DLB_ReturnClaimedCpus &&
         DLB_ReturnClaimedCpu &&
         DLB_ClaimCpus &&
         DLB_CheckCpuAvailability;
#endif

   if ( _status.dlb_enabled )
      DLB_Init();

   _status.initialized = _status.dlb_enabled || _status.block_enabled;
}

void ResourceManager::finalize( void )
{
   _status.initialized = false;

   if ( _status.dlb_enabled )
      DLB_Finalize();
}

/* Check the availabilty of resources
   and claim my cpus if necessary
*/
void ResourceManager::acquireResourcesIfNeeded ( void )
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t ready_tasks_key = ID->getEventKey("concurrent-tasks"); )

   ThreadTeam *team;

   if ( !_status.initialized ) return;
   if ( !(team = getMyThreadSafe()->getTeam()) ) return;

   if ( _status.is_malleable ) {
      /* OmpSs*/

      int ready_tasks = team->getSchedulePolicy().getPotentiallyParallelWDs();

      if ( ready_tasks > 0 ){

         NANOS_INSTRUMENT( nanos_event_value_t ready_tasks_value = (nanos_event_value_t) ready_tasks )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &ready_tasks_key, &ready_tasks_value); )

         int needed_resources = ready_tasks - team->getFinalSize();

         LockBlock Lock( _lock );
         if ( needed_resources > 0 ) {
            if ( _status.dlb_enabled ) {
               //If ready tasks > num threads I claim my cpus being used by someone else
               const cpu_set_t& process_mask = sys.getCpuProcessMask();
               const cpu_set_t& active_mask = sys.getCpuActiveMask();
               cpu_set_t mine_and_active;
               CPU_AND( &mine_and_active, &process_mask, &active_mask );
               if ( !CPU_EQUAL(&mine_and_active, &process_mask) ) {
                  // Only claim if some of my CPUs are not active
                  DLB_ClaimCpus( needed_resources );
               }

               //If ready tasks > num threads I check if there are available cpus
               needed_resources = ready_tasks - team->getFinalSize();
               if ( needed_resources > 0 ){
                  DLB_UpdateResources_max( needed_resources );
               }
            } else {
               // Iterate over default cpus not running and wake them up if needed
               const cpu_set_t& process_mask = sys.getCpuProcessMask();
               cpu_set_t new_active_cpus = sys.getCpuActiveMask();
               for (int i=0; i<_max_cpus; i++) {
                  if ( CPU_ISSET( i, &process_mask) && !CPU_ISSET( i, &new_active_cpus ) ) {
                     CPU_SET( i, &new_active_cpus );
                     if ( --ready_tasks == 0 )
                        break;
                     if ( CPU_EQUAL( &process_mask, &new_active_cpus ) )
                        break;
                  }
               }
               sys.setCpuActiveMask( &new_active_cpus );
            }
         }
      }

   } else {
      /* OpenMP */
      LockBlock Lock( _lock );
      if ( _status.dlb_enabled ) {
         DLB_UpdateResources();
      } else {
         const cpu_set_t& process_mask = sys.getCpuProcessMask();
         sys.setCpuActiveMask( &process_mask );
      }
   }
}


/* When there is no work to do, release my cpu and go to sleep
   This function only has effect in DLB when using policy: auto_LeWI_mask
*/
void ResourceManager::releaseCpu( void )
{
   if ( !_status.initialized ) return;
   if ( !_status.is_malleable ) return;
   if ( !_status.block_enabled ) return;
   if ( !getMyThreadSafe()->getTeam() ) return;
   if ( getMyThreadSafe()->isSleeping() ) return;

   int my_cpu = getMyThreadSafe()->getCpuId();

   LockBlock Lock( _lock );

   // Do not release if this CPU is the last active within the process_mask
   cpu_set_t mine_and_active;
   CPU_AND( &mine_and_active, &(sys.getCpuProcessMask()), &(sys.getCpuActiveMask()) );
   if ( CPU_COUNT( &mine_and_active ) == 1 && CPU_ISSET( my_cpu, &mine_and_active ) )
      return;

   if ( _status.dlb_enabled ) {
      DLB_ReleaseCpu( my_cpu );
   } else {
      cpu_set_t new_active_cpus = sys.getCpuActiveMask();
      ensure( CPU_ISSET(my_cpu, &new_active_cpus), "Trying to release a non active CPU" );
      CPU_CLR( my_cpu, &new_active_cpus );
      sys.setCpuActiveMask( &new_active_cpus );
   }
}

/* Only called by master thread
   Check if any of our cpus have been claimed by its owner
*/
void ResourceManager::returnClaimedCpus( void )
{
   if ( !_status.initialized ) return;
   if ( !_status.is_malleable ) return;
   if ( !_status.dlb_enabled ) return;
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

/* Only useful for external slave (?) threads
   Check if my cpu has been claimed
   and return it if necesary by going to sleep
   This function only has effect in DLB when using policy: auto_LeWI_mask
*/
void ResourceManager::returnMyCpuIfClaimed( void )
{
   if ( !_status.initialized ) return;
   if ( !_status.is_malleable ) return;
   if ( !_status.dlb_enabled ) return;

   // Return if my cpu belongs to the default mask
   const cpu_set_t& process_mask = sys.getCpuProcessMask();
   int my_cpu = getMyThreadSafe()->getCpuId();
   if ( CPU_ISSET( my_cpu, &process_mask ) )
      return;

   if ( !getMyThreadSafe()->isSleeping() ) {
      LockBlock Lock( _lock );
      DLB_ReturnClaimedCpu( my_cpu );
   }
}

/* When waking up to check if the my cpu is "really" free
   while it is not free, wait
*/
void ResourceManager::waitForCpuAvailability( void )
{
   if ( !_status.initialized ) return;
   if ( !_status.dlb_enabled ) return;

   int cpu = getMyThreadSafe()->getCpuId();
   CPU_SET( cpu, &_waiting_cpus );
   while ( !lastActiveThread() && !DLB_CheckCpuAvailability(cpu) ) {
      // Sleep and Yield the thread to reduce cycle consumption
      OS::nanosleep( NANOS_RM_YIELD_SLEEP_NS );
      sched_yield();
   }
   CPU_CLR( cpu, &_waiting_cpus );
}

bool ResourceManager::lastActiveThread( void )
{
   if ( !_status.initialized ) return true;

   // We omit the test if the cpu does not belong to my process_mask
   int my_cpu = getMyThreadSafe()->getCpuId();
   if ( !CPU_ISSET( my_cpu, &(sys.getCpuProcessMask()) ) ) return false;

   LockBlock Lock( _lock );
   cpu_set_t mine_and_active;
   CPU_AND( &mine_and_active, &(sys.getCpuProcessMask()), &(sys.getCpuActiveMask()) );
   return ( CPU_COUNT( &mine_and_active ) == 1 && CPU_ISSET( my_cpu, &mine_and_active ) );
}
