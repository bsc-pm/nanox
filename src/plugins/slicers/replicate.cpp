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

   nanos_ws_desc_t *wsd_current = *(( nanos_ws_desc_t ** )work.getData());

   int i = myThread->getTeam()->getFinalSize() - 1;

   BaseThread *thread = &(myThread->getTeam()->getThread(i));
   if ( thread == myThread ) {
      i--;
      thread = &(myThread->getTeam()->getThread(i));
   }

   BaseThread *last_thread = thread;
   i--;
   while ( i >= 0 ) {
      thread = &(myThread->getTeam()->getThread(i));
      if ( thread != myThread ) {
         WorkDescriptor *slice = NULL;
         sys.duplicateWD( &slice, &work );
         sys.setupWD(*slice, &work );
         slice->tieTo( *thread );
         ((WorkSharing *)(wsd_current->ws))->duplicateWS( wsd_current, ( nanos_ws_desc_t ** ) slice->getData() );
         thread->addNextWD( slice );
      }
      i--;
   }

   // Converting original workdescriptor to a regular tied one and submitting it
   work.convertToRegularWD();
   work.tieTo( *last_thread );
   ((WorkSharing *)(wsd_current->ws))->duplicateWS( wsd_current, ( nanos_ws_desc_t ** ) work.getData() );
   last_thread->addNextWD( (WorkDescriptor *) &work );

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
   *slice = wd;
   return true;
}

namespace ext {

class SlicerReplicatePlugin : public Plugin {
   public:
      SlicerReplicatePlugin () : Plugin("Slicer which replicate a given wd into a list of a given threads",1) {}
      ~SlicerReplicatePlugin () {}

      virtual void config( Config& cfg ) {}

      void init ()
      {
         sys.registerSlicer("replicate", NEW SlicerReplicate() );	
      }
};

} // namespace ext
} // namespace nanos

DECLARE_PLUGIN( "placeholder-name", nanos::ext::SlicerReplicatePlugin );
