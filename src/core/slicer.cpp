/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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

#include "nanos-int.h"
#include "slicer.hpp"
#include "debug.hpp"
#include "system.hpp"

using namespace nanos;


/*! \brief Submit a RepeatN slicedWD
 *
 *  This function submits a RepeatN slicedWD using the Scheduler
 *
 */
void SlicerRepeatN::submit ( WD &work )
{
   debug0 ( "Using sliced work descriptor: RepeatN" );
   Scheduler::submit ( work );
}

/* \brief Dequeue a RepeatN SlicedWD
 *
 *  This function dequeues a RepeantN SlicedWD returning true if there
 *  will be no more slices to manage (i.e. this is the last chunk to
 *  execute. The received paramenter wd has to be associated with a 
 *  SlicerRepeatN and SlicerDataRepeatN objects.
 * 
 *  \param [in] wd is the former WorkDescriptor
 *  \param [out] slice is the next portion to execute
 *
 *  \return true if there are no more slices in the former wd, false otherwise
 */
bool SlicerRepeatN::dequeue ( SlicedWD *wd, WorkDescriptor **slice)
{

   debug0 ( "Dequeueing sliced work: RepeatN start" );

   int n = ((SlicerDataRepeatN *)(wd->getSlicerData()))->decN();

   if ( n > 0 ) 
   {
      debug0 ( "Dequeueing sliced work: keeping former wd" );
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

void SlicerDynamicFor::submit ( WD &work )
{
   debug0 ( "Using sliced work descriptor: Dynamic For" );
   Scheduler::submit ( work );
}

bool SlicerDynamicFor::dequeue ( SlicedWD *wd, WorkDescriptor **slice )
{
   int lower, upper, step;

   if ( ((SlicerDataDynamicFor *)(wd->getSlicerData()))->getNextIters( &lower, &upper, &step ) ) 
   {
      // last iters
      debug0 ( "Dequeueing sliced work: using former wd (final), loop={l:" << lower << " u:" << upper <<" s:" << step << "}");
      *slice = wd;
      ((nanos_loop_info_t *)((*slice)->getData()))->lower = lower;
      ((nanos_loop_info_t *)((*slice)->getData()))->upper = upper;
      ((nanos_loop_info_t *)((*slice)->getData()))->step = step;
      return true;
   }
   else
   {
      // no last iters
      debug0 ( "Dequeueing sliced work: keeping former wd loop={l:" << lower << " u:" << upper << " s:" << step << "}");
      sys.duplicateWD( slice, wd );
      ((nanos_loop_info_t *)((*slice)->getData()))->lower = lower;
      ((nanos_loop_info_t *)((*slice)->getData()))->upper = upper;
      ((nanos_loop_info_t *)((*slice)->getData()))->step = step;
      return false;
   }
}
