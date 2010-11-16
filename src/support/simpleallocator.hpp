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

#ifndef _NANOS_SIMPLEALLOCATOR
#define _NANOS_SIMPLEALLOCATOR

#include <stdint.h>
#include <map>

namespace nanos {

   /*! \brief Simple memory allocator to manage a given contiguous memory area
    */
   class SimpleAllocator
   {
      private:
         typedef std::map < uintptr_t, size_t > SegmentMap;

         SegmentMap _allocatedChunks;
         SegmentMap _freeChunks;

      public:
         SimpleAllocator( uintptr_t baseAddress, size_t len );

         void * allocate( size_t len );
         size_t free( void *address );
         void * reallocate( void *address, size_t len );
   };

}
#endif /* _NANOS_SIMPLEALLOCATOR */
