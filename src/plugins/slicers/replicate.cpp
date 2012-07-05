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

  /* Getting thread map info: thread vector is only guaranteed to be availabe during submit phase, so
   * if the map is needed further than that it will be needed a copy */
   BaseThread **threads = (BaseThread**) (*(( nanos_ws_desc_t ** )work.getData()))->threads;
   int n = (*(( nanos_ws_desc_t ** )work.getData()))->nths;

   nanos_ws_desc_t *wsd_current = *(( nanos_ws_desc_t ** )work.getData());

   n--;

   // Creating (n-1) tied workdescriptors and submiting them
   while ( n > 0 ) {
      WorkDescriptor *slice = NULL;
      sys.duplicateWD( &slice, &work );
      sys.setupWD(*slice, &work);
      slice->tieTo( *threads[n] );
      ((WorkSharing *)(wsd_current->ws))->duplicateWS( wsd_current, ( nanos_ws_desc_t ** ) slice->getData() );
      if ( threads[n]->setNextWD( slice ) == false ) Scheduler::submit ( *slice );
      n--;
   }

   // Converting original workdescriptor to a regular tied one and submiting it
   work.convertToRegularWD();
   work.tieTo( *threads[0] );
   ((WorkSharing *)(wsd_current->ws))->duplicateWS( wsd_current, ( nanos_ws_desc_t ** ) work.getData() );
   if ( threads[0]->setNextWD( (WorkDescriptor *) &work) == false ) {
     Scheduler::submit ( work );
   }
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
