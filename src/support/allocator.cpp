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

#include "allocator.hpp"
#include "basethread.hpp"

using namespace nanos;

Allocator *nanos::allocator;

size_t Allocator::_headerSize = NANOS_ALIGNED_MEMORY_OFFSET( 0, sizeof(Allocator::ObjectHeader), 16 );

Allocator & nanos::getAllocator ( void )
{
   if (!allocator) {
      allocator = (Allocator *) malloc(sizeof(Allocator));
      if ( allocator == NULL ) throw(NANOS_ENOMEM);
      new (allocator) Allocator();
   }

   BaseThread *my_thread = getMyThreadSafe();
   if ( my_thread == NULL ) return *allocator;
   else return my_thread->getAllocator();
}

void * Allocator::Arena::allocate ( void )
{
   unsigned int obj;

   if ( !_free ) return NULL;

   for ( obj = 0 ; obj < numObjects ; obj++ ) {
      if ( _bitmap[obj]._bit ) break;
   }

   if (obj == numObjects) {
      _free = false;
      return NULL; 
   }

   _bitmap[obj]._bit = false;

   return (void *) &_arena[obj*_objectSize];
}


