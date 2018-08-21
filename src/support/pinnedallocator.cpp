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

#include <iostream>
#include <cstring>
#include "pinnedallocator.hpp"
#include "gpudevice_decl.hpp"

using namespace nanos;


void * CUDAPinnedMemoryManager::allocate( size_t size )
{
   return GPUDevice::allocatePinnedMemory( size );
}

void CUDAPinnedMemoryManager::free( void * address )
{
   return GPUDevice::freePinnedMemory( address );
}


bool PinnedAllocator::isPinned( void * address, size_t size )
{
   PinnedMemoryMap::iterator it = _pinnedChunks.lower_bound( address );

   // Perfect match, check size
   if ( it->first == address ) {
      if ( it->second >= size ) return true;

      // Size is bigger than pinned area
      return false;
   }

   // address is lower than any other pinned address
   if ( it == _pinnedChunks.begin() ) return false;

   // It is an intermediate region, check it fits into a pinned area
   it--;

   if ( ( it->first < address ) && ( ( ( size_t ) it->first + it->second ) >= ( ( size_t ) address + size ) ) )
      return true;

   return false;
}

void PinnedAllocator::printPinnedMemoryMap()
{
   std::cout << "PINNED MEMORY CHUNKS" << std::endl;
   for (PinnedMemoryMap::iterator it = _pinnedChunks.begin(); it != _pinnedChunks.end(); it++ ) {
      std::cout << "|... ";
      std::cout << it->first << " @ " << it->second;
      std::cout << " ...";
   }
   std::cout << "|" << std::endl;
}


