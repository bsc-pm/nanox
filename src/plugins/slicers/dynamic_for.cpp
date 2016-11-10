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
#include "smpdd.hpp"

namespace nanos {
namespace ext {

class SlicerDynamicFor: public Slicer
{
   private:
   public:
      // constructor
      SlicerDynamicFor ( ) { }

      // destructor
      ~SlicerDynamicFor ( ) { }

      // headers (implemented below)
      void submit ( WorkDescriptor & work ) ;
      bool dequeue(nanos::WorkDescriptor* wd, nanos::WorkDescriptor** slice);
};

void SlicerDynamicFor::submit ( WorkDescriptor &work )
{
   debug0 ( "Using sliced work descriptor: Dynamic For" );

   nanos_loop_info_t *nli = (nanos_loop_info_t *) work.getData();

   //! Normalize Chunk size
   nli->chunk = (1 > nli->chunk)? 1 : nli->chunk;

   work.untie();
   Scheduler::submit ( work );
}

bool SlicerDynamicFor::dequeue(nanos::WorkDescriptor* wd, nanos::WorkDescriptor** slice)
{
   bool retval = false;

   nanos_loop_info_t *nli = ( nanos_loop_info_t * ) wd->getData();

   //! Compute next (chunk) lower bound
   int64_t _upper = nli->lower + nli->chunk * nli->step - nli->step;

   //! Computing empty iteration spaces in order to avoid infinite task generation
   bool empty = (( nli->step > 0 ) && (nli->lower > nli->upper )) ? true : false;
   empty = empty || (( nli->step < 0 ) && (nli->lower < nli->upper )) ? true : false;

   if ( (_upper >= nli->upper) || empty ) {
      *slice = wd; retval = true;
   } else {
      WorkDescriptor *nwd = NULL;
      sys.duplicateWD( &nwd, wd );
      nwd->untie();
      nli->lower = _upper + nli->step;

      nli = ( nanos_loop_info_t * ) nwd->getData();
      nli->upper = _upper;
      sys.setupWD(*nwd, wd );

      *slice = nwd;
   }

   return retval; 
}

class SlicerDynamicForPlugin : public Plugin {
   public:
      SlicerDynamicForPlugin () : Plugin("Slicer for Loops using a dynamic policy",1) {}
      ~SlicerDynamicForPlugin () {}

      virtual void config( Config& cfg ) {}

      void init ()
      {
         sys.registerSlicer("dynamic_for", NEW SlicerDynamicFor() );	
      }
};

} // namespace ext
} // namespace nanos

DECLARE_PLUGIN("slicer-dynamic_for",nanos::ext::SlicerDynamicForPlugin);
