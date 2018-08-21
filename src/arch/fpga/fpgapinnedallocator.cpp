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

#include "fpgapinnedallocator.hpp"
#include "debug.hpp"

using namespace nanos;

void * FPGAPinnedAllocator::allocate( size_t size )
{
   void * newBuffer;
   xdma_buf_handle bufferHandle;
   xdma_status status;
   status = xdmaAllocateKernelBuffer( &newBuffer, &bufferHandle, size );
   if ( status != XDMA_SUCCESS ) {
      warning0( "Could not allocate pinned memory" );
      //TODO: do something intelligent in this case
      return NULL;
   }
   _lock.acquire();
   _pinnedChunks[ newBuffer ] = size;
   _handleMap[ newBuffer ] = bufferHandle;
   _lock.release();

   return newBuffer;
}

void FPGAPinnedAllocator::free( void * address )
{
   xdma_buf_handle bHandle;
   bHandle = _handleMap[ address ];
   xdmaFreeKernelBuffer( address, bHandle );
   _lock.acquire();
   _pinnedChunks.erase( address );
   _handleMap.erase( address );
   _lock.release();
}

//size should not be needed to determine the base address of a pointer
void * FPGAPinnedAllocator::getBasePointer( void * address, size_t size )
{
   PinnedMemoryMap::iterator it = _pinnedChunks.lower_bound( address );

   // Perfect match, check size
   if ( it->first == address ) {
      if ( it->second >= size ) return it->first;

      // Size is bigger than pinned area
      return NULL;
   }

   // address is lower than any other pinned address
   if ( it == _pinnedChunks.begin() ) return NULL;

   // It is an intermediate region, check it fits into a pinned area
   it--;

   if ( ( it->first < address ) && ( ( ( size_t ) it->first + it->second ) >= ( ( size_t ) address + size ) ) )
      return it->first;

   return NULL;
}

xdma_buf_handle FPGAPinnedAllocator::getBufferHandle( void * address )
{
   return _handleMap[ address ];
}

void FPGAPinnedAllocator::addBufferHandle( void * address, xdma_buf_handle handle )
{
   _handleMap[ address ] = handle;
}

void FPGAPinnedAllocator::delBufferHandle( void * address )
{
   _handleMap.erase( address );
}
