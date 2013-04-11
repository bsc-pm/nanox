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
      void submit ( SlicedWD & work ) ;
      bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) ;
};

void SlicerReplicate::submit ( SlicedWD &work )
{
   debug0 ( "Using sliced work descriptor: Replicate" );

   nanos_ws_desc_t *wsd_current = *(( nanos_ws_desc_t ** )work.getData());

   int i = myThread->getTeam()->size() - 1;

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

/* \brief Dequeue a Replicate SlicedWD
 *
 *  \param [in] wd is the former WorkDescriptor
 *  \param [out] slice is the next portion to execute
 *
 *  \return true if there are no more slices in the former wd, false otherwise
 */
bool SlicerReplicate::dequeue ( SlicedWD *wd, WorkDescriptor **slice)
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

nanos::ext::SlicerReplicatePlugin NanosXPlugin;
