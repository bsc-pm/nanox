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

#include <string.h>
#include "processingelement.hpp"
#include "debug.hpp"
#include "schedule.hpp"
#include "copydata.hpp"

using namespace nanos;

BaseThread& ProcessingElement::startWorker ( SchedulingGroup *sg )
{
   WD & worker = getWorkerWD();
   return startThread( worker,sg );
}

BaseThread & ProcessingElement::startThread ( WD &work, SchedulingGroup *sg )
{
   BaseThread &thread = createThread( work );

   if ( sg ) sg->addMember( thread );

   thread.start();

   _threads.push_back( &thread );

   return thread;
}

BaseThread & ProcessingElement::associateThisThread ( SchedulingGroup *sg, bool untieMain )
{
   WD & worker = getMasterWD();
   
   BaseThread &thread = createThread( worker );

   if ( sg ) sg->addMember( thread );

   thread.associate();

   if ( !untieMain ) {
      worker.tieTo(thread);
   }

   return thread;
}

void ProcessingElement::stopAll ()
{
   ThreadList::iterator it;
   BaseThread *thread;

   for ( it = _threads.begin(); it != _threads.end(); it++ ) {
      thread = *it;
      thread->stop();
      thread->join();
      if ( thread->hasTeam() )
         thread->leaveTeam();
   }
}

void* ProcessingElement::getAddress( WorkDescriptor &wd, void* tag, nanos_sharing_t sharing )
{
   void *actualTag = (void *) ( sharing == NX_PRIVATE ? (char *)wd.getData() + (unsigned long)tag : tag );
   return actualTag;
}

void ProcessingElement::copyTo( WorkDescriptor& wd, void* dst, void *tag, nanos_sharing_t sharing, size_t size )
{
   void *actualTag = (void *) ( sharing == NX_PRIVATE ? (char *)wd.getData() + (unsigned long)tag : tag );
   // FIXME: should this be done by using the local copeir of the device?
   memcpy( dst, actualTag, size );
}

