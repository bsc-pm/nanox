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

/*! \file nanos_finders.cpp
 *  \brief 
 */
#include "nanos.h"
#include "debug.hpp"
#include "system.hpp"
#include "plugin.hpp"
#include "instrumentationmodule_decl.hpp"

/*! \defgroup capi_finders Finder services.
 *  \ingroup capi
 */
/*! \addtogroup capi_finders
 *  \{
 */

using namespace nanos;

/*! \brief Find a slicer giving a label id
 *
 *  \sa Slicers
 */
NANOS_API_DEF(nanos_slicer_t, nanos_find_slicer, ( const char * label ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","find_slicer",NANOS_RUNTIME) );

   nanos_slicer_t slicer;
   
   try {
      std::string plugin = std::string(label);
      slicer = sys.getSlicer ( plugin );
      if ( slicer == NULL ) {
         if ( !sys.loadPlugin( "slicer-" + plugin )) fatal0( "Could not load " + plugin + "slicer" );
         slicer = sys.getSlicer ( plugin );
      }

   } catch ( nanos_err_t e) {
      return ( nanos_slicer_t ) NULL;
   }
   return slicer;
}

/*! \brief Find a worksharing giving a label id
 *
 *  \sa WorkSharing
 */
NANOS_API_DEF(nanos_ws_t, nanos_find_worksharing, ( const char * label ))
{
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","",NANOS_RUNTIME) ); //FIXME: to register new event

   nanos_ws_t ws;
   try {
      std::string plugin = std::string(label);
      ws = sys.getWorkSharing ( plugin );
      if ( ws == NULL ) {
         if ( !sys.loadPlugin( "worksharing-" + plugin )) fatal0( "Could not load " + plugin + "worksharing" );
         ws = sys.getWorkSharing ( plugin );
      }

   } catch ( nanos_err_t e) {
      return ( nanos_ws_t ) NULL;
   }
   return ws;
}

/*!
 * \}
 */ 
