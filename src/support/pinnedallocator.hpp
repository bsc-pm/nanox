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

#ifndef _NANOS_PINNEDALLOCATOR
#define _NANOS_PINNEDALLOCATOR

#include <stdint.h>
#include <map>
#include "pinnedallocator_decl.hpp"
#include "atomic.hpp"

namespace nanos {

PinnedMemoryManager::PinnedMemoryManager() {}
PinnedMemoryManager::~PinnedMemoryManager() {}


CUDAPinnedMemoryManager::CUDAPinnedMemoryManager() {}
CUDAPinnedMemoryManager::~CUDAPinnedMemoryManager() {}


PinnedAllocator::PinnedAllocator( PinnedMemoryManager * manager) : _pinnedChunks(), _manager( manager ), _lock() {}

void * PinnedAllocator::allocate( size_t size )
{
   void * addr = _manager->allocate( size );

   _lock.acquire();
   _pinnedChunks[ addr ] = size;
   _lock.release();

   return addr;
}

void PinnedAllocator::free( void * address )
{
   _lock.acquire();
   _pinnedChunks.erase( address );
   _lock.release();
   _manager->free( address );
}

} // namespace nanos

#endif /* _NANOS_PINNEDALLOCATOR */
