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

extern "C" {
   void DLB_UpdateResources_max( int max_resources ) __attribute__(( weak ));
   void DLB_UpdateResources( void ) __attribute__(( weak ));
   void DLB_ReturnClaimedCpus( void ) __attribute__(( weak ));
   int DLB_ReleaseCpu ( int cpu ) __attribute__(( weak ));
   int DLB_ReturnClaimedCpu ( int cpu ) __attribute__(( weak ));
   void DLB_ClaimCpus (int cpus) __attribute__(( weak ));
   int DLB_CheckCpuAvailability ( int cpu ) __attribute__(( weak ));
}

namespace nanos {
namespace ResourceManager {
   namespace {
      Lock      _lock;
      cpu_set_t _running_cpus;
      cpu_set_t _default_cpus;
   }
}}

using namespace nanos;


void ResourceManager::init( void )
{
   LockBlock Lock( _lock );
   sys.getCpuMask( &_running_cpus );
   sys.getCpuMask( &_default_cpus );
   ensure( CPU_COUNT(&_running_cpus)>0, "Resource Manager: empty mask" );
}

/* Check the availabilty of resources
   and claim my cpus if necessary
*/
void ResourceManager::acquireResourcesIfNeeded ( void )
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t ready_tasks_key = ID->getEventKey("concurrent-tasks"); )

   ThreadTeam *team = getMyThreadSafe()->getTeam();
   if ( !team )
      return;

   if ( sys.getPMInterface().isMalleable() ) {
      /* OmpSs*/

      int ready_tasks = team->getSchedulePolicy().getPotentiallyParallelWDs();

      if ( ready_tasks > 0 ){

         NANOS_INSTRUMENT( nanos_event_value_t ready_tasks_value = (nanos_event_value_t) ready_tasks )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &ready_tasks_key, &ready_tasks_value); )

         int needed_resources = ready_tasks - team->getFinalSize();
         if ( needed_resources > 0 ){
            LockBlock Lock( _lock );

            if ( sys.dlbEnabled() && DLB_UpdateResources ) {
               //If ready tasks > num threads I claim my cpus being used by somebodyels
               DLB_ClaimCpus( needed_resources );

               //If ready tasks > num threads I check if there are available cpus
               needed_resources = ready_tasks - team->getFinalSize();
               if ( needed_resources > 0 ){
                  DLB_UpdateResources_max( needed_resources );
               }
            }
            sys.getCpuMask( &_running_cpus );
            ensure( CPU_COUNT(&_running_cpus)>0, "Resource Manager: empty mask" );
         }
      }

   } else {
      /* OpenMP */
      if ( sys.dlbEnabled() && DLB_UpdateResources ) {
         DLB_UpdateResources();
      } else {
         // wakeup all in stored cpus
      }
   }
}


/* When there is not work to do
   release my cpu and go to sleep
   This function only has effect in DLB when using policy: auto_LeWI_mask
*/
void ResourceManager::releaseCpu( void )
{
   if ( !getMyThreadSafe()->getTeam() )
      return;

   bool release = true;
   if ( sys.getPMInterface().isMalleable() && !getMyThreadSafe()->isSleeping() ){
      int my_cpu = getMyThreadSafe()->getCpuId();

      // TODO: use acquire/release instead of block?
      LockBlock Lock( _lock );

      if ( CPU_COUNT(&_running_cpus) > 1 ) {

         if ( sys.dlbEnabled() && DLB_ReleaseCpu ) {
            release = DLB_ReleaseCpu( my_cpu );
         }

         if ( release ) {
            CPU_CLR( my_cpu, &_running_cpus );
            ensure( CPU_COUNT(&_running_cpus)>0, "Resource Manager: empty mask" );
      myThread->sleep();
            //_lock.release();
            //myThread->sleep();
            //myThread->wait();
         //} else {
            //_lock.release();
         }
      }
      //else {
         //release = false;
      //}
   }
   //if ( release ) {
      //myThread->wait();
      //}
}

/* Only called by master thread
   Check if any of our cpus have been claimed by its owner
   return it if necesary notifying the corresponding thread
*/
void ResourceManager::returnClaimedCpus( void )
{
   if ( sys.dlbEnabled() && DLB_ReturnClaimedCpus && getMyThreadSafe()->isMainThread() && sys.getPMInterface().isMalleable() ){
      DLB_ReturnClaimedCpus();
   }

   LockBlock Lock( _lock );
   sys.getCpuMask( &_running_cpus );
   ensure( CPU_COUNT(&_running_cpus)>0, "Resource Manager: empty mask" );
}

/* Only useful for external slave (?) threads
   Check if my cpu has been claimed
   and return it if necesary by going to sleep
   This function only has effect in DLB when using policy: auto_LeWI_mask
*/
void ResourceManager::returnMyCpuIfClaimed( void )
{
   // Return if my cpu belongs to the default mask
   int my_cpu = getMyThreadSafe()->getCpuId();
   if ( CPU_ISSET( my_cpu, &_default_cpus ) )
      return;

   bool released = false;
   if ( sys.dlbEnabled() && DLB_ReturnClaimedCpu && sys.getPMInterface().isMalleable() && !getMyThreadSafe()->isSleeping() ){
      LockBlock Lock( _lock );
      released = DLB_ReturnClaimedCpu( my_cpu );
      if ( released ) {
         myThread->sleep();
         // We can't clear it since DLB could have switch this cpu for another one
         //CPU_CLR( my_cpu, &_running_cpus );
         sys.getCpuMask( &_running_cpus );
         ensure( CPU_COUNT(&_running_cpus)>0, "Resource Manager: empty mask" );
      }
   }
}

/* When waking up to check if the my cpu is "really" free
   while it is not free, wait
*/
void ResourceManager::waitForCpuAvailability( void )
{
   if ( sys.dlbEnabled() && DLB_CheckCpuAvailability ){
      int cpu = getMyThreadSafe()->getCpuId();
      while ( CPU_COUNT(&_running_cpus) > 1 && !DLB_CheckCpuAvailability(cpu) )
         sched_yield();
      /*      if ((myThread->getTeam()->getSchedulePolicy().fixme_getNumConcurrentWDs()==0) && DLB_ReleaseCpu(cpu)){
            myThread->sleep();
            break;
            }*/
   }
}

bool ResourceManager::lastOne( void )
{
   LockBlock Lock( _lock );
   return ( CPU_COUNT(&_running_cpus) == 1 );
}
