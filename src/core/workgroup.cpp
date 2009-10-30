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

#include "workgroup.hpp"
#include "atomic.hpp"
#include "schedule.hpp"

using namespace nanos;

Atomic<int> WorkGroup::atomicSeed( 0 );

void WorkGroup::addWork ( WorkGroup &work )
{
   components++;
   work.addToGroup( *this );
}

void WorkGroup::addToGroup ( WorkGroup &parent )
{
   partOf.push_back( &parent );
}

void WorkGroup::exitWork ( WorkGroup &work )
{
   components--;
}

void WorkGroup::sync ()
{
   phase_counter++;
   //TODO: block and switch

   while ( phase_counter < components );

   //TODO: reinit phase_counter
}

void WorkGroup::waitCompletation ()
{
   Scheduler::blockOnCondition<int>( &components.override(),0 );
}

void WorkGroup::done ()
{
   for ( ListOfWGs::iterator it = partOf.begin();
         it != partOf.end();
         it++ ) {
      ( *it )->exitWork( *this );
      //partOf.erase(it);
   }
}

WorkGroup::~WorkGroup ()
{
   done();
}
