/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#include "plugin.hpp"
#include "slicer.hpp"
#include "system.hpp"
#include "worksharing_decl.hpp"

namespace nanos {

class SlicerReplicate: public Slicer
{
   private:
   public:
      // constructor
      SlicerReplicate ( ) { }

      // destructor
      ~SlicerReplicate ( ) { }

      // headers (implemented below)
      void submit ( WorkDescriptor & work ) ;
      bool dequeue ( WorkDescriptor *wd, WorkDescriptor **slice ) ;
};

void SlicerReplicate::submit ( WorkDescriptor &work )
{
   debug0 ( "Using sliced work descriptor: Replicate" );

   nanos_ws_desc_t *wsd = *(nanos_ws_desc_t **)work.getData();
   WorkSharing *ws = (WorkSharing *)wsd->ws;
   if (ws->instanceOnCreation()) {
      /* WS static */
      ThreadTeam *team = myThread->getTeam();
      int team_size = team->getFinalSize();
      for (int i=0; i<team_size; ++i) {
         BaseThread &thread = team->getThread(i);
         if (&thread != myThread) {
            /* duplicate */
            WorkDescriptor *slice = NULL;
            sys.duplicateWD( &slice, &work );
            sys.setupWD( *slice, &work );
            slice->tieTo( thread );
            ws->duplicateWS( wsd, (nanos_ws_desc_t **)slice->getData() );
            thread.addNextWD( slice );
         }
      }
      work.convertToRegularWD();
      work.tieTo( *myThread );
      myThread->addNextWD( &work );
   } else {
      /* WS dynamic or guided */
      /* Submit as a regular untied WD */
      work.untie();
      Scheduler::submit ( work );

      /* Enable CPUs from the process mask if needed */
      sys.getThreadManager()->acquireDefaultCPUs( ws->getItemsLeft(wsd) - 1 );
   }
}

/* \brief Dequeue a Replicate WorkDescriptor
 *
 *  \param [in] wd is the former WorkDescriptor
 *  \param [out] slice is the next portion to execute
 *
 *  \return true if there are no more slices in the former wd, false otherwise
 */
bool SlicerReplicate::dequeue ( WorkDescriptor *wd, WorkDescriptor **slice)
{
   debug0 ( "Dequeueing sliced work: Replicate start" );

   /* dequeue_wd will only be true if there is only one chunk left in the WS */
   nanos_ws_desc_t *wsd = *(nanos_ws_desc_t **)wd->getData();
   WorkSharing *ws = (WorkSharing *)wsd->ws;
   bool dequeue_wd = ws->getItemsLeft( wsd ) <= 1;

   if (dequeue_wd) {
      /* This is the last slice of the WD */
      *slice = wd;
   } else {
      /* Otherwise, duplicate */
      WorkDescriptor *nwd = NULL;
      sys.duplicateWD( &nwd, wd );
      sys.setupWD( *nwd, wd );
      ws->duplicateWS( wsd, (nanos_ws_desc_t **)nwd->getData() );
      *slice = nwd;
   }

   /* Tie slice always */
   (*slice)->tieTo( *myThread );

   return dequeue_wd;
}

namespace ext {

class SlicerReplicatePlugin : public Plugin {
   public:
      SlicerReplicatePlugin () : Plugin("Slicer for WorkSharings", 1) {}
      ~SlicerReplicatePlugin () {}

      virtual void config( Config& cfg ) {}

      void init ()
      {
         sys.registerSlicer("replicate", NEW SlicerReplicate() );
      }
};

} // namespace ext
} // namespace nanos

DECLARE_PLUGIN( "slicer-replicate", nanos::ext::SlicerReplicatePlugin );
