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
      cpu_set_t _running_cpus;
      cpu_set_t _waiting_cpus;
      cpu_set_t _default_cpus;
   }
}}

using namespace nanos;


void ResourceManager::init( void )
{
   LockBlock Lock( _lock );
   sys.getCpuActiveMask( &_running_cpus );
   sys.getCpuProcessMask( &_default_cpus );
   CPU_ZERO( &_waiting_cpus );
   ensure( CPU_COUNT(&_running_cpus)>0, "Resource Manager: empty mask" );

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

   // DLB is only manually finalized when Nanos has --enable-block (No MPI)
   if ( _status.dlb_enabled && _status.block_enabled )
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
         if ( needed_resources > 0 ){
            LockBlock Lock( _lock );

            if ( _status.dlb_enabled ) {
               //If ready tasks > num threads I claim my cpus being used by somebodyels
               DLB_ClaimCpus( needed_resources );

               //If ready tasks > num threads I check if there are available cpus
               needed_resources = ready_tasks - team->getFinalSize();
               if ( needed_resources > 0 ){
                  DLB_UpdateResources_max( needed_resources );
               }
            } else {
               // Iterate over default cpus not running and wake them up if needed
               for (int i=0; i<CPU_SETSIZE; i++) {
                  if ( CPU_ISSET( i, &_default_cpus) && !CPU_ISSET( i, &_running_cpus ) ) {
                     CPU_SET( i, &_running_cpus );
                     if ( --ready_tasks == 0 )
                        break;
                     if ( CPU_EQUAL( &_default_cpus, &_running_cpus ) )
                        break;
                  }
               }
                sys.setCpuActiveMask( &_running_cpus );
            }
            sys.getCpuActiveMask( &_running_cpus );
            ensure( CPU_COUNT(&_running_cpus)>0, "Resource Manager: empty mask" );
         }
      }

   } else {
      /* OpenMP */
      if ( _status.dlb_enabled ) {
         LockBlock Lock( _lock );
         DLB_UpdateResources();
         sys.getCpuActiveMask( &_running_cpus );
         ensure( CPU_COUNT(&_running_cpus)>0, "Resource Manager: empty mask" );
      } else {
         LockBlock Lock( _lock );
         memcpy( &_running_cpus, &_default_cpus, sizeof(cpu_set_t) );
         sys.setCpuActiveMask( &_default_cpus );
      }
   }
}


/* When there is not work to do
   release my cpu and go to sleep
   This function only has effect in DLB when using policy: auto_LeWI_mask
*/
void ResourceManager::releaseCpu( void )
{
   if ( !_status.initialized ) return;
   if ( !_status.is_malleable ) return;
   if ( !_status.block_enabled ) return;
   if ( !getMyThreadSafe()->getTeam() ) return;
   if ( getMyThreadSafe()->isSleeping() ) return;

   bool release = true;
   int my_cpu = getMyThreadSafe()->getCpuId();

   {
      LockBlock Lock( _lock );

      if ( CPU_COUNT(&_running_cpus) > 1 ) {

         if ( _status.dlb_enabled ) {
            release = DLB_ReleaseCpu( my_cpu );
         }

         if ( release ) {
            CPU_CLR( my_cpu, &_running_cpus );
            ensure( CPU_COUNT(&_running_cpus)>0, "Resource Manager: empty mask" );
            myThread->sleep();
         }
      }
   }
   // wait() only after the lock is released
   if ( getMyThreadSafe()->isSleeping() ) {
      myThread->wait();
   }
}

/* Only called by master thread
   Check if any of our cpus have been claimed by its owner
   return it if necesary notifying the corresponding thread
*/
void ResourceManager::returnClaimedCpus( void )
{
   if ( !_status.initialized ) return;
   if ( !_status.is_malleable ) return;
   if ( !_status.dlb_enabled ) return;
   if ( !getMyThreadSafe()->isMainThread() ) return;

   DLB_ReturnClaimedCpus();
   LockBlock Lock( _lock );
   sys.getCpuActiveMask( &_running_cpus );
   ensure( CPU_COUNT(&_running_cpus)>0, "Resource Manager: empty mask" );
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
   int my_cpu = getMyThreadSafe()->getCpuId();
   if ( CPU_ISSET( my_cpu, &_default_cpus ) )
      return;

   if ( !getMyThreadSafe()->isSleeping() ){
      LockBlock Lock( _lock );
      if ( DLB_ReturnClaimedCpu( my_cpu ) ) {
         myThread->sleep();
         // We can't clear it since DLB could have switched this cpu for another one
         //CPU_CLR( my_cpu, &_running_cpus );
         sys.getCpuActiveMask( &_running_cpus );
         ensure( CPU_COUNT(&_running_cpus)>0, "Resource Manager: empty mask" );
      }
   }

   // Go to sleep inmediately
   if ( myThread->isSleeping() )
      myThread->wait();
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
   while ( !lastOne() && !DLB_CheckCpuAvailability(cpu) )
      // Sleep the thread for a while to reduce the cycles consumption, then yield
      OS::nanosleep( NANOS_RM_YIELD_SLEEP_NS );
      sched_yield();
   /*      if ((myThread->getTeam()->getSchedulePolicy().fixme_getNumConcurrentWDs()==0) && DLB_ReleaseCpu(cpu)){
         myThread->sleep();
         break;
         }*/
   CPU_CLR( cpu, &_waiting_cpus );
}

bool ResourceManager::lastOne( void )
{
   if ( !_status.initialized )
      return false;

   LockBlock Lock( _lock );
   return ( CPU_COUNT(&_running_cpus) - CPU_COUNT(&_waiting_cpus) <= 1 );
}
