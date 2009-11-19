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

#include "slicer.hpp"
#include "debug.hpp"
#include "system.hpp"

using namespace nanos;


void SlicerStatic::submit ( WD &work )
{
   debug0 ( "Using sliced work descriptor: static" );
   Scheduler::submit ( work );
}

bool SlicerStatic::dequeue( WD *wd, WD **slice)
{
   *slice = wd;
   return true;
/*
   StaticSlicerData *sd = (StaticSlicerData *) candidate->getSlicerData();

   int id = myThread->getTeamId();
   int low,upper =  ...

   WD *wd = new WD(....);

   result = wd;

   if (sd->last) return true;
   return false;
*/
}

