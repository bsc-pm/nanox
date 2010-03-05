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

#include "nanos_pe.h"
#include "basethread.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "workdescriptor.hpp"
#include "plugin.hpp"

using namespace nanos;

nanos_err_t nanos_get_addr ( void * tag, nanos_sharing_t sharing, void **addr )
{
   sys.getInstrumentor()->enterRuntime();

   nanos_wd_t cwd = myThread->getCurrentWD();
   WD *wd = ( WD * )cwd;

   ProcessingElement *pe = myThread->runningOn();
   *addr = pe->getAddress( *wd, tag, sharing ); //FIXME , SHARING );

   sys.getInstrumentor()->leaveRuntime();

   return NANOS_OK;
}

nanos_err_t nanos_copy_value ( void * dst, void *tag, nanos_sharing_t sharing, size_t size )
{
   sys.getInstrumentor()->enterRuntime();

   nanos_wd_t cwd = myThread->getCurrentWD();
   WD *wd = ( WD * )cwd;

   ProcessingElement *pe = myThread->runningOn();
   pe->copyTo( *wd, dst, tag, sharing, size ); //FIXME , SHARING );

   sys.getInstrumentor()->leaveRuntime();

   return NANOS_OK;
}

