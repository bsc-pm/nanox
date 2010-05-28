#include "plugin.hpp"
#include "slicer.hpp"
#include "system.hpp"

namespace nanos {

class SlicerCompoundWD: public Slicer
{
   private:
   public:
      // constructor
      SlicerCompoundWD ( ) : Slicer() { }

      // destructor
      ~SlicerCompoundWD ( ) { }

      // headers (implemented below)
      void submit ( SlicedWD & work ) ;
      bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) ;
      void *getSpecificData() const;
      static void executeWDs ( nanos_compound_wd_data_t *data );
};

void SlicerCompoundWD::submit ( SlicedWD &work )
{
   debug ( "Using sliced work descriptor: CompoundWD" );

   nanos_compound_wd_data_t *data = (nanos_compound_wd_data_t *) work.getData();
   WorkDescriptor *slice;

   /* As the wd's has not been submitted we need to configure it */
   for ( int i = 0; i < data->nsect; i++) {
      slice = ((WorkDescriptor**)data->lwd)[i];
      slice->setParent ( &work );                                                                                                          
      slice->setDepth( work.getDepth() +1 );
   }

   Scheduler::submit ( work );
}

/* \brief Dequeue a SlicerCompoundWD WD
 *
 *  This function dequeues a SlicerCompoundWD returning true if there
 *  will be no more slices to manage (i.e. this is the last section to
 *  execute.
 *
 *  \param [in] wd is the original WorkDescriptor
 *  \param [out] slice is the next portion to execute
 *
 *  \return true if there are no more slices in the former wd, false otherwise
 */
bool SlicerCompoundWD::dequeue ( SlicedWD *wd, WorkDescriptor **slice )
{
   /* Get compound wd data */
   nanos_compound_wd_data_t *data = (nanos_compound_wd_data_t *) wd->getData();

   /* If we have executed all wd's or we want to serialize FIXME(true) them */
   if ( ( data->nsect == 1 ) || sys.getNumWorkers() == 1 ) {
      *slice = wd;
      return true;
   }

   /* Pre-decrement nsect and get corresponding wd */
   *slice = ((WorkDescriptor**)data->lwd)[--(data->nsect)];

   return false;
}

void *SlicerCompoundWD::getSpecificData ( ) const
{
   return (void *) executeWDs;
}

void SlicerCompoundWD::executeWDs ( nanos_compound_wd_data_t *data )
{
   WorkDescriptor *slice;

   for ( int i = 0; i < data->nsect; i++ ) {
      slice = ((WorkDescriptor**)data->lwd)[i];
      Scheduler::inlineWork( slice );
   }

}

namespace ext {

class SlicerCompoundWDPlugin : public Plugin {
   public:
      SlicerCompoundWDPlugin () : Plugin("Slicer which aggregates several wd's",1) { }
      ~SlicerCompoundWDPlugin () { }

      virtual void config( Config& config ) { }

      void init () { sys.registerSlicer("compound_wd", new SlicerCompoundWD() ); }
};

} // namespace ext
} // namespace nanos

nanos::ext::SlicerCompoundWDPlugin NanosXPlugin;

