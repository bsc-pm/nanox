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

#include "dependableobjectwd.hpp"
#include "workdescriptor.hpp"
#include "schedule.hpp"

using namespace nanos;

/*! \brief Submits WorkDescriptor when dependencies are satisfied
 */
void DOSubmit::dependenciesSatisfied ( )
{
     _submittedWD->submit();
}

/*! \brief whether the DO gets blocked and no more dependencies can
 *  be submitted until it is satisfied.
 */
bool DOWait::waits()
{
   return true;
}


/*! \brief Initialise wait condition
 */
void DOWait::init()
{
   _depsSatisfied = false;
}

/*! \brief Wait method blocks execution untill dependencies are satisfied
 */
void DOWait::wait ( )
{
     _syncCond.wait();
}

/*! \brief Unblock method when dependencies are satisfied
 */
void DOWait::dependenciesSatisfied ( )
{
   _syncCond.signal();
}

