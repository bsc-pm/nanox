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

#include "processingelement.hpp"
#include "debug.hpp"
#include "schedule.hpp"

using namespace nanos;

BaseThread& ProcessingElement::startWorker ( SchedulingGroup *sg )
{
   WD & master = getWorkerWD();
   return startThread( master,sg );
}

BaseThread & ProcessingElement::startThread ( WD &work, SchedulingGroup *sg )
{
   BaseThread &thread = createThread( work );

   if ( sg ) sg->addMember( thread );

   thread.start();

   _threads.push_back( &thread );

   return thread;
}

BaseThread & ProcessingElement::associateThisThread ( SchedulingGroup *sg )
{
   WD & master = getMasterWD();
   BaseThread &thread = createThread( master );

   if ( sg ) sg->addMember( thread );

   thread.associate();

   return thread;
}

void ProcessingElement::stopAll ()
{
   ThreadList::iterator it;

   for ( it = _threads.begin(); it != _threads.end(); it++ ) {
      ( *it )->stop();
      ( *it )->join();
   }
}
