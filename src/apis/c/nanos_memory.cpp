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
/*! \file nanos_memory.cpp
 *  \brief 
 */
#include "nanos.h"
#include "allocator.hpp"
#include "memtracker.hpp"
#include "instrumentation_decl.hpp"
#include "instrumentationmodule_decl.hpp"

#include <cstring>

/*! \defgroup capi_mem Memory services.
 *  \ingroup capi
 */
/*! \addtogroup capi_mem
 *  \{
 */

NANOS_API_DEF(nanos_err_t, nanos_malloc, ( void **p, size_t size, const char *file, int line ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","malloc",NANOS_RUNTIME ) );

   try 
   {
#if defined(NANOS_DEBUG_ENABLED) && defined(NANOS_MEMTRACKER_ENABLED)
      if ( line != 0 ) *p = nanos::getMemTracker().allocate( size, file, line );
      else *p = nanos::getMemTracker().allocate( size );
#else
      *p = nanos::getAllocator().allocate ( size );
#endif
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_free, ( void *p ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","free",NANOS_RUNTIME ) );

   try 
   {
#if defined(NANOS_DEBUG_ENABLED) && defined(NANOS_MEMTRACKER_ENABLED)
      nanos::getMemTracker().deallocate( p );
#else
      nanos::getAllocator().deallocate ( p );
#endif
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(void, nanos_free0, ( void *p ))
{
   nanos_free(p);
}

NANOS_API_DEF(nanos_err_t, nanos_memcpy, (void *dest, const void *src, size_t n))
{
    std::memcpy(dest, src, n);
    return NANOS_OK;
}
/*!
 * \}
 */ 
