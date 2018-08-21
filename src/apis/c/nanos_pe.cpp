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

/*! \file nanos_pe.cpp
 *  \brief 
 */
#include "nanos.h"
#include "basethread.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "workdescriptor.hpp"
#include "plugin.hpp"
#include "instrumentationmodule_decl.hpp"

using namespace nanos;

/*! \defgroup capi_pe Processing Element services.
 *  \ingroup capi
 */
/*! \addtogroup capi_pe
 *  \{
 */

NANOS_API_DEF(nanos_err_t, nanos_get_addr, ( nanos_copy_id_t copy_id, void **addr, nanos_wd_t cwd ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_addr",NANOS_RUNTIME) );

   WD *wd = ( WD * )cwd;

   //*addr = (void *) wd->_ccontrol.getAddress( copy_id );
   *addr = (void *) wd->_mcontrol.getAddress( copy_id );

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_copy_value, ( void *dst, nanos_copy_id_t copy_id, nanos_wd_t cwd ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","copy_value",NANOS_RUNTIME) );

   std::cerr << __FUNCTION__ << ": Not supported." << std::endl;
   //WD *wd = ( WD * )cwd;
   //CopyData &cd = wd->getCopies()[copy_id];

   //ProcessingElement *pe = myThread->runningOn();
   //pe->copyTo( *wd, dst, cd.getAddress(), cd.getSharing(), cd.getSize() );

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_get_node_num, ( unsigned int * node ))
{
   *node = sys.getNetwork()->getNodeNum();
   return NANOS_OK;
}

NANOS_API_DEF(int, nanos_get_num_nodes, ())
{
   return sys.getNetwork()->getNumNodes();
}
