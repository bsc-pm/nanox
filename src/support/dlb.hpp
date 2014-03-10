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

   inline void dlb_returnCpusIfNeeded ( void )
   {
      if ( sys.dlbEnabled() && DLB_ReturnClaimedCpus && getMyThreadSafe()->getId() == 0 && sys.getPMInterface().isMalleable() ){
         if ( getMyThreadSafe()->getId() == 0 ){
        //    DLB_ReturnClaimedCpus();
        ;
         }else{
            int myCpu=getMyThreadSafe()->getCpuId();
            DLB_ReturnClaimedCpu(myCpu);
            //bool released=DLB_ReturnClaimedCpu(myCpu);
/*            if (released){
               myThread->sleep();
            }*/
         }
      }
   }

   inline void dlb_updateAvailableCpus ( void )
   {
      if ( sys.dlbEnabled() && DLB_UpdateResources_max ){
         //If I'm not the master thread I only release my cpu if it has been claimed
         if ( getMyThreadSafe()->getId() != 0 ) {
            int myCpu=getMyThreadSafe()->getCpuId();
            bool released=DLB_ReturnClaimedCpu(myCpu);
            if (released){
               myThread->sleep();
            }

         //If I'm the master thread I'll return first the Claimed cpus
         }else{
//            DLB_ReturnClaimedCpus();
         
            if ( sys.getPMInterface().isMalleable() ) {
               //If ready tasks > num threads I claim my cpus being used by somebodyelse
               int needed_resources = sys.getSchedulerStats().getReadyTasks() - sys.getNumThreads();
               if ( needed_resources > 0){
                  DLB_ClaimCpus(needed_resources);
               }
               //If ready tasks > num threads I check if there are available cpus
               needed_resources = sys.getSchedulerStats().getReadyTasks() - sys.getNumThreads();
               if ( needed_resources > 0 ){
                  DLB_UpdateResources_max( needed_resources );
               }
            } else {
               DLB_UpdateResources();
            }
         }

      }

   }

   inline bool dlb_releaseMyCpu( void ){
  
      bool released=0;
      int myCpu=getMyThreadSafe()->getCpuId();
      int me =getMyThreadSafe()->getId();
      if ( sys.dlbEnabled() && DLB_ReleaseCpu && sys.getPMInterface().isMalleable() && me != 0 && !getMyThreadSafe()->isTaggedToSleep() ){
         released=DLB_ReleaseCpu(myCpu);
      }
      return released;
   }

   inline bool dlb_returnMyCpuIfClaimed( void ){
      bool released=0;
      int myCpu=getMyThreadSafe()->getCpuId();
      int me =getMyThreadSafe()->getId();
      if ( sys.dlbEnabled() && DLB_ReturnClaimedCpu && sys.getPMInterface().isMalleable() && me != 0 && !getMyThreadSafe()->isTaggedToSleep() ){
         released=DLB_ReturnClaimedCpu(myCpu);
      }
      return released;
      
   }

   inline void dlb_checkCpuAvailability (){
      if ( sys.dlbEnabled() && DLB_CheckCpuAvailability ){
         int cpu=getMyThreadSafe()->getCpuId();
         while (!DLB_CheckCpuAvailability(cpu))
            sched_yield();
      }
   }
}
#endif
