#include "plugin.hpp"
#include "slicer.hpp"
#include "system.hpp"

namespace nanos {

class SlicerCompoundWD: public Slicer
{
   private:
   public:
      // constructor
      SlicerCompoundWD ( ) { }

      // destructor
      ~SlicerCompoundWD ( ) { }

      // headers (implemented below)
      void submit ( SlicedWD & work ) ;
      bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) ;
      void *getSpecificData() const;
      static void executeWDs ( WD *lwd[] );
};

void SlicerCompoundWD::submit ( SlicedWD &work )
{
   debug0 ( "Using sliced work descriptor: CompoundWD" );
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
bool SlicerCompoundWD::dequeue ( SlicedWD *wd, WorkDescriptor **slice)
{
   /* Get next wd index */
   int n = ((SlicerDataCompoundWD *)(wd->getSlicerData()))->getNextIndex();

   /* If next index to execute is -1, there is no more wd to execute */
   if ( n == -1 ) {
      *slice = wd;
      return true;
   }

   /* Get next WD to execute */
   *slice = ( (WorkDescriptor**) (wd->getData()) )[n];

   /* As *slice has not submited we need to configure it */
   (*slice)->setParent ( wd );                                                                                                          
   (*slice)->setDepth( wd->getDepth() +1 );                                                                                                     
 
   return false;
}

void *SlicerCompoundWD::getSpecificData ( ) const
{
   return (void *) executeWDs;
}

void SlicerCompoundWD::executeWDs ( WD *lwd[] )
{


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

