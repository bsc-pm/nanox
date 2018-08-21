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

namespace nanos {

class SlicerRepeatN: public Slicer
{
   private:
   public:
      // constructor
      SlicerRepeatN ( ) { }

      // destructor
      ~SlicerRepeatN ( ) { }

      // headers (implemented below)
      void submit ( WorkDescriptor & work ) ;
      bool dequeue ( WorkDescriptor *wd, WorkDescriptor **slice ) ;
};

void SlicerRepeatN::submit ( WorkDescriptor &work )
{
   debug0 ( "Using sliced work descriptor: RepeatN" );
   Scheduler::submit ( work );
}

/* \brief Dequeue a RepeatN WorkDescriptor
 *
 *  This function dequeues a RepeantN WorkDescriptor returning true if there
 *  will be no more slices to manage (i.e. this is the last chunk to
 *  execute. The received paramenter wd has to be associated with a
 *  SlicerRepeatN.
 *
 *  \param [in] wd is the former WorkDescriptor
 *  \param [out] slice is the next portion to execute
 *
 *  \return true if there are no more slices in the former wd, false otherwise
 */
bool SlicerRepeatN::dequeue ( WorkDescriptor *wd, WorkDescriptor **slice)
{

   debug0 ( "Dequeueing sliced work: RepeatN start" );

   int n = --((( nanos_repeat_n_info_t * )wd->getData())->n);

   if ( n > 0 )
   {
      debug0 ( "Dequeueing sliced work: keeping former wd" );
      *slice = NULL;
      sys.duplicateWD( slice, wd );
      sys.setupWD(**slice, wd);

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

class SlicerRepeatNPlugin : public Plugin {
   public:
      SlicerRepeatNPlugin () : Plugin("Slicer for repeating n times a given wd",1) {}
      ~SlicerRepeatNPlugin () {}

      virtual void config( Config& cfg ) {}

      void init ()
      {
         sys.registerSlicer("repeat_n", NEW SlicerRepeatN() );	
      }
};

} // namespace ext
} // namespace nanos

DECLARE_PLUGIN("slicer-repeat_n",nanos::ext::SlicerRepeatNPlugin);
