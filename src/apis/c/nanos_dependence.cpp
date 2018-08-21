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

/*! \file nanos_dependences.cpp
 *  \brief 
 */
#include "nanos.h"
#include "system.hpp"
#include "instrumentationmodule_decl.hpp"
#include "basethread.hpp"
#include "workdescriptor.hpp"

/*! \defgroup capi_dependence Dependence services.
 *  \ingroup capi
 */

/*! \addtogroup capi_dependence
 *  \{
 */

using namespace nanos;

//! \brief Release all current WorkDescriptor dependences
NANOS_API_DEF(nanos_err_t, nanos_dependence_release_all, ( void ) )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","dependence_release_all",NANOS_RUNTIME) );
   try {
      WD *parent = NULL, *wd = myThread->getCurrentWD();
      if ( wd ) parent = wd->getParent();
      if ( parent ) parent->workFinished( *wd );
   } catch ( nanos_err_t e) {
      return e;                                                                                                                          
   }
   return NANOS_OK;
}

/*! \brief Returns if there are any pendant write for a given addr
 *
 *  \param [out] res is the result
 *  \param [in] addr is the related address
 */
NANOS_API_DEF(nanos_err_t, nanos_dependence_pendant_writes, ( bool *res, void *addr ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","dependence_pendant_writes",NANOS_RUNTIME) );
   try {
      *res = ( bool ) sys.haveDependencePendantWrites( addr );
   } catch ( nanos_err_t e) {
      return e;                                                                                                                          
   }
   return NANOS_OK;
}

//! \brief Create a new dependence between two different WD's
//!
//! This function is not thread safe, so user must guarantee thread safety and avoid
//! race condition due a parallel dependence creation and execution dependence releases
//!
//! \param [in] pred is the predecessor work descriptor
//! \param [in] succ is the successor work descriptor
NANOS_API_DEF(nanos_err_t, nanos_dependence_create, ( nanos_wd_t pred, nanos_wd_t succ ) )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","dependence_create", NANOS_RUNTIME) );
   try {
      sys.createDependence( (WD *) pred, (WD *) succ );
   } catch ( nanos_err_t e) {
      return e;                                                                                                                          
   }
   return NANOS_OK;
}
/*!
 * \}
 */ 
