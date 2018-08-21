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

/*! \file nanos_memory.cpp
 *  \brief 
 */
#include "nanos.h"
#include "allocator.hpp"
#include "memtracker.hpp"
#include "osallocator_decl.hpp"
#include "instrumentation_decl.hpp"
#include "instrumentationmodule_decl.hpp"

#include <cstring>

/*! \defgroup capi_mem Memory services.
 *  \ingroup capi
 */
/*! \addtogroup capi_mem
 *  \{
 */

using namespace nanos;

NANOS_API_DEF(nanos_err_t, nanos_malloc, ( void **p, size_t size, const char *file, int line ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","malloc",NANOS_RUNTIME ) );

   try 
   {
#if defined(NANOS_DEBUG_ENABLED) && defined(NANOS_MEMTRACKER_ENABLED)
      if ( line != 0 ) *p = nanos::getMemTracker().allocate( size, file, line );
      else *p = nanos::getMemTracker().allocate( size );
#elif defined(NANOS_ENABLE_ALLOCATOR)
      *p = nanos::getAllocator().allocate ( size );
#else
      *p = malloc(size);
#endif
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_memalign, ( void **p, size_t size, const char *file, int line ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","memalign",NANOS_RUNTIME ) );

   try 
   {
      nanos::OSAllocator tmp_allocator;
      *p = tmp_allocator.allocate ( size );
   } catch ( nanos_err_t e ) {
      return e;
   }

   return NANOS_OK;
}


NANOS_API_DEF(nanos_err_t, nanos_cmalloc, ( void **p, size_t size, unsigned int node, const char *file, int line ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","cmalloc",NANOS_RUNTIME ) );

   if ( node < nanos::sys.getNetwork()->getNumNodes() ) {
      try 
      {
         nanos::OSAllocator tmp_allocator;
         if ( node == 0 ) {
            *p = tmp_allocator.allocate ( size );
         } else {
            *p = tmp_allocator.allocate_none( size );
         }
         nanos::sys.registerNodeOwnedMemory(node, *p, size);
      } catch ( nanos_err_t e ) {
         return e;
      }

      return NANOS_OK;
   } else {
      return NANOS_INVALID_PARAM;
   }
}

NANOS_API_DEF(nanos_err_t, nanos_cmalloc_2dim_distributed, ( void **p, size_t rows, size_t cols, size_t elem_size, unsigned int start_node, size_t num_nodes, const char *file, int line ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","cmalloc",NANOS_RUNTIME ) );

   if ( start_node > 0 && num_nodes > 0 &&
         start_node < nanos::sys.getNetwork()->getNumNodes() &&
         start_node + (num_nodes-1) < nanos::sys.getNetwork()->getNumNodes() ) {
      try 
      {
         size_t size = cols * rows * elem_size;
         nanos::OSAllocator tmp_allocator;
         *p = tmp_allocator.allocate_none( size );
         nanos::global_reg_t reg = nanos::sys._registerMemoryChunk_2dim(*p, rows, cols, elem_size);
         nanos::sys._distributeObject( reg, start_node, num_nodes );
      } catch ( nanos_err_t e ) {
         return e;
      }

      return NANOS_OK;
   } else {
      return NANOS_INVALID_PARAM;
   }
}

NANOS_API_DEF(nanos_err_t, nanos_stick_to_producer, ( void *p, size_t size ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","stick_to_producer",NANOS_RUNTIME ) );

   try 
   {
      nanos::sys.stickToProducer(p, size);
   } catch ( nanos_err_t e ) {
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
#elif defined(NANOS_ENABLE_ALLOCATOR)
      nanos::getAllocator().deallocate ( p );
#else
      free ( p );
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

NANOS_API_DEF(nanos_err_t, nanos_register_object, (int num_objects, nanos_copy_data_t *obj))
{
   nanos::sys.registerObject( num_objects, obj );
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_unregister_object, (int num_objects, void *base_addresses))
{
   nanos::sys.unregisterObject( num_objects, base_addresses );
   return NANOS_OK;
}

/*!
 * \}
 */ 
