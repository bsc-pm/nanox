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
}

namespace nanos {

   inline void dlb_returnCpusIfNeeded ( void )
   {
      if ( sys.dlbEnabled() && DLB_ReturnClaimedCpus && getMyThreadSafe()->getId() == 0 && sys.getPMInterface().isMalleable() )
         DLB_ReturnClaimedCpus();
   }

   inline void dlb_updateAvailableCpus ( void )
   {
      if ( sys.dlbEnabled() && DLB_UpdateResources_max && getMyThreadSafe()->getId() == 0 ) {
            DLB_ReturnClaimedCpus();

         if ( sys.getPMInterface().isMalleable() ) {
            int needed_resources = sys.getSchedulerStats().getReadyTasks() - sys.getNumThreads();
            if ( needed_resources > 0 )
               DLB_UpdateResources_max( needed_resources );

         } else {
            DLB_UpdateResources();
         }


      }

   }
}
#endif
