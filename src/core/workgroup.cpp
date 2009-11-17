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

Atomic<int> WorkGroup::_atomicSeed( 0 );

void WorkGroup::addWork ( WorkGroup &work )
{
   _components++;
   work.addToGroup( *this );
}

void WorkGroup::addToGroup ( WorkGroup &parent )
{
   _partOf.push_back( &parent );
}

void WorkGroup::exitWork ( WorkGroup &work )
{
   _components--;
}

void WorkGroup::sync ()
{
   _phaseCounter++;
   //TODO: block and switch

   while ( _phaseCounter < _components );

   //TODO: reinit phase_counter
}

void WorkGroup::waitCompletation ()
{
   Scheduler::blockOnCondition<int>( &_components.override(),0 );
}

void WorkGroup::done ()
{
   for ( WGList::iterator it = _partOf.begin();
         it != _partOf.end();
         it++ ) {
      ( *it )->exitWork( *this );
      //partOf.erase(it);
   }
}

WorkGroup::~WorkGroup ()
{
   done();
}

void WorkGroup::dependenciesSatisfied()
{
}

