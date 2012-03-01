#include "plugin.hpp"
#include "slicer.hpp"
#include "system.hpp"

namespace nanos {

class SlicerExpandTeam: public Slicer
{
   private:
   public:
      // constructor
      SlicerExpandTeam ( ) { }

      // destructor
      ~SlicerExpandTeam ( ) { }

      // headers (implemented below)
      void submit ( SlicedWD & work ) ;
      bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) ;
};

void SlicerExpandTeam::submit ( SlicedWD &work )
{
   debug0 ( "Using sliced work descriptor: ExpandTeam" );
   Scheduler::submit ( work );
}

/* \brief Dequeue a ExpandTeam SlicedWD
 *
 *  This function dequeues a RepeantN SlicedWD returning true if there
 *  will be no more slices to manage (i.e. this is the last chunk to
 *  execute. The received paramenter wd has to be associated with a
 *  SlicerExpandTeam.
 *
 *  \param [in] wd is the former WorkDescriptor
 *  \param [out] slice is the next portion to execute
 *
 *  \return true if there are no more slices in the former wd, false otherwise
 */
bool SlicerExpandTeam::dequeue ( SlicedWD *wd, WorkDescriptor **slice)
{

   debug0 ( "Dequeueing sliced work: ExpandTeam start" );

   int n = --((( nanos_repeat_n_info_t * )wd->getData())->n);

   if ( n > 0 )
   {
      debug0 ( "Dequeueing sliced work: keeping former wd" );
      *slice = NULL;
      sys.duplicateWD( slice, wd );

      return false;
   }
   else
   {
      debug0 ( "Dequeueing sliced work: using former wd (final)" );
      *slice = wd;
      return true;
   }
}

namespace ext {

class SlicerExpandTeamPlugin : public Plugin {
   public:
      SlicerExpandTeamPlugin () : Plugin("Slicer for expanding current wd to other non implicit threads into the team",1) {}
      ~SlicerExpandTeamPlugin () {}

      virtual void config( Config& cfg ) {}

      void init ()
      {
         sys.registerSlicer("expand_team", NEW SlicerExpandTeam() );	
      }
};

} // namespace ext
} // namespace nanos

nanos::ext::SlicerExpandTeamPlugin NanosXPlugin;
