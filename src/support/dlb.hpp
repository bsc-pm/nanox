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

#ifndef _NANOS_DLB
#define _NANOS_DLB


using namespace nanos;

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

   /* Only called by master thread
      Check if any of our cpus have been claimed by its owner
      return it if necesary notifying the corresponding thread
   */
   inline void dlb_returnClaimedCpus ( void )
   {
      if ( sys.dlbEnabled() && DLB_ReturnClaimedCpus && getMyThreadSafe()->getId() == 0 && sys.getPMInterface().isMalleable() ){
         DLB_ReturnClaimedCpus();
      }
   }

   /* Only called by slave threads
      Check if my cpu has been claimed 
      and return it if necesary by going to sleep
      This function only has effect in DLB when using policy: auto_LeWI_mask
   */
   inline bool dlb_returnMyCpuIfClaimed( void ){
      bool released=0;
      int myCpu=getMyThreadSafe()->getCpuId();
      int me =getMyThreadSafe()->getId();
      if ( sys.dlbEnabled() && DLB_ReturnClaimedCpu && sys.getPMInterface().isMalleable() && me != 0 && !getMyThreadSafe()->isSleeping() ){
         released=DLB_ReturnClaimedCpu(myCpu);
         if (released){
            myThread->sleep();
            myThread->getTeam()->decreaseFinalSize();
         }
      }
      return released;
   }
   

   /* Only called by master thread
      Check the availabilty of resources
      and claim my cpus if necessary
   */
   inline void dlb_updateAvailableCpus ( void )
   {
      if ( sys.dlbEnabled() && DLB_UpdateResources && getMyThreadSafe()->getId() == 0){
         
            if ( sys.getPMInterface().isMalleable() ) {
               NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

               NANOS_INSTRUMENT ( static nanos_event_key_t ready_tasks_key  = ID->getEventKey("concurrent-tasks"); )
               
               //If ready tasks > num threads I claim my cpus being used by somebodyels
               int ready_tasks= myThread->getTeam()->getSchedulePolicy().getPotentiallyParallelWDs();

               NANOS_INSTRUMENT ( nanos_event_value_t ready_tasks_value = (nanos_event_value_t) ready_tasks )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &ready_tasks_key, &ready_tasks_value); )

               int needed_resources = ready_tasks - myThread->getTeam()->getFinalSize();

               if ( needed_resources > 0){
                  DLB_ClaimCpus(needed_resources);
               }
               //If ready tasks > num threads I check if there are available cpus
               needed_resources = ready_tasks - getMyThreadSafe()->getTeam()->getFinalSize();

               if ( needed_resources > 0 ){
                  DLB_UpdateResources_max( needed_resources );
               }
            } else {
               DLB_UpdateResources();
            }
         }
   }

   /* Only called by slave threads
      When there is not work to do
      release my cpu and go to sleep
      This function only has effect in DLB when using policy: auto_LeWI_mask
   */
   inline bool dlb_releaseMyCpu( void ){
  
      bool released=0;
      int myCpu=getMyThreadSafe()->getCpuId();
      int me =getMyThreadSafe()->getId();
      if ( sys.dlbEnabled() && DLB_ReleaseCpu && sys.getPMInterface().isMalleable() && me != 0 && !getMyThreadSafe()->isSleeping() ){
         released=DLB_ReleaseCpu(myCpu);
         if (released){
            myThread->sleep();
            myThread->getTeam()->decreaseFinalSize();
         }
      }
      return released;
   }

   /* Only called by slave threads 
      When waking up to check if the my cpu is "really" free
      while it is not free, wait.
   */
   inline void dlb_checkCpuAvailability (){
      int goToSleep=0;
      if ( sys.dlbEnabled() && DLB_CheckCpuAvailability ){
         int cpu=getMyThreadSafe()->getCpuId();
         while ((goToSleep==0) && (!DLB_CheckCpuAvailability(cpu)))
            sched_yield();
      /*      if ((myThread->getTeam()->getSchedulePolicy().fixme_getNumConcurrentWDs()==0) && DLB_ReleaseCpu(cpu)){
                  myThread->sleep();
                  goToSleep=1;
            }*/
      }
   }
}
#endif
