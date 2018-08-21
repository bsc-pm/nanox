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

/*! \file nanos_worksharing.cpp
 *  \brief 
 */
#include "nanos.h"
#include "worksharing_decl.hpp"

/*! \defgroup capi_ws Worksharing services.
 *  \ingroup capi
 */
/*! \addtogroup capi_ws
 *  \{
 */

using namespace nanos;

NANOS_API_DEF(nanos_err_t, nanos_worksharing_create, ( nanos_ws_desc_t **wsd, nanos_ws_t ws, nanos_ws_info_t *info,  bool *b ) )
{
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","",NANOS_RUNTIME) ); //FIXME: To register new event

   try {
      if ( b ) *b = ((WorkSharing *) ws)->create( wsd, info );
      else ((WorkSharing *) ws)->create( wsd, info );
   } catch ( nanos_err_t e) {
      return e;
   }
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_worksharing_next_item, ( nanos_ws_desc_t *wsd, nanos_ws_item_t *wsi ))
{
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","",NANOS_RUNTIME) ); //FIXME: To register new event

   try {
      ((WorkSharing *) wsd->ws)->nextItem( wsd, wsi );
   } catch ( nanos_err_t e) {
      return e;
   }
   return NANOS_OK;

}
/*!
 * \}
 */ 
