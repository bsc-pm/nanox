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

#include <utility>
#include "dependenciesdomain.hpp"
#include "commutationdepobj.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "dataaccess.hpp"


namespace nanos {

Atomic<int> DependenciesDomain::_atomicSeed( 0 );
Atomic<int> DependenciesDomain::_tasksInGraph( 0 );
Lock DependenciesDomain::_lock;

using namespace dependencies_domain_internal;

void DependenciesDomain::increaseTasksInGraph( size_t num )
{
   NANOS_INSTRUMENT(lock();)
   NANOS_INSTRUMENT(nanos_event_value_t tasks = (nanos_event_value_t)(_tasksInGraph += num);)
   NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("graph-size");)
   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents(1, &key, (nanos_event_value_t *) &tasks );)
   NANOS_INSTRUMENT(unlock();)
}

void DependenciesDomain::decreaseTasksInGraph( size_t num )
{
   NANOS_INSTRUMENT(lock();)
   NANOS_INSTRUMENT(nanos_event_value_t tasks = (nanos_event_value_t)(_tasksInGraph-=num);)
   //NANOS_INSTRUMENT(int tasks = --_tasksInGraph;)
   NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("graph-size");)
   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents(1, &key, (nanos_event_value_t *) &tasks );)
   NANOS_INSTRUMENT(unlock();)
}
} // namespace nanos
