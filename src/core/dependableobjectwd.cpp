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
#include "synchronizedcondition.hpp"

using namespace nanos;

void DOSubmit::dependenciesSatisfied ( )
{
     _submittedWD->submit();
}

bool DOWait::waits()
{
   return true;
}


void DOWait::init()
{
   _depsSatisfied = false;
}

void DOWait::wait ( )
{
     _syncCond.wait();
}

void DOWait::dependenciesSatisfied ( )
{
   _depsSatisfied = true;
   _syncCond.signal();
}

