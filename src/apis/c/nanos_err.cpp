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

/*! \file nanos_err.cpp
 *  \brief 
 */
#include "nanos.h"
#include <stdlib.h>
#include <stdio.h>

#if defined(NANOS_DEBUG_ENABLED) && defined(NANOS_INSTRUMENTATION_ENABLED)
   char nanos_mode[] = "instrumentation-debug";
#elif defined(NANOS_DEBUG_ENABLED) && !defined(NANOS_INSTRUMENTATION_ENABLED)
   char nanos_mode[] = "debug";
#elif !defined(NANOS_DEBUG_ENABLED) && defined(NANOS_INSTRUMENTATION_ENABLED)
   char nanos_mode[] = "instrumentation";
#elif !defined(NANOS_DEBUG_ENABLED) && !defined(NANOS_INSTRUMENTATION_ENABLED)
   char nanos_mode[] = "performance";
#endif

NANOS_API_DEF(char *, nanos_get_mode, ( void ))
{
   return nanos_mode;
}

NANOS_API_DEF(void, nanos_handle_error, ( nanos_err_t err ))
{
   switch ( err ) {

      default:

      case NANOS_UNKNOWN_ERR:
         fprintf( stderr,"Nanox: Unknown NANOS error detected\n" );
         break;
      case NANOS_UNIMPLEMENTED:
         fprintf( stderr,"Nanox: Requested NANOS service not implemented\n" );
         break;
      case NANOS_ENOMEM:
         fprintf( stderr,"Nanox: Cannot allocate enough memory to run the program\n" );
         break;
      case NANOS_INVALID_PARAM:
         fprintf( stderr, "Nanox: invalid parameter\n" );
         break;
      case NANOS_INVALID_REQUEST:
         fprintf( stderr, "Nanox: invalid request\n" );
         break;
   }

   abort();
}
