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
   int n = ((SlicerDataCompoundWD *)(wd->getSlicerData()))->getNextIndex();
   *slice = ( (WorkDescriptor**)(wd->getData()) )[n];
   if ( n == 0 ) return true;
   else return false;
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
