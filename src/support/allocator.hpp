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
#ifndef _NANOS_ALLOCATOR_HPP
#define _NANOS_ALLOCATOR_HPP

#include <vector>
#include <cstdlib>
#include <cstring>
#include "malign.hpp"
#include <iostream>
#include "allocator_decl.hpp"

inline size_t Allocator::getObjectSize ()
{
   return _objectSize;
}

void * Arena::allocate ( void )
{
   unsigned int obj;
   for ( obj = 0 ; obj < numObjects ; obj++ )
      if ( _bitmap[obj]._bit ) break;

      if (obj == numObjects) {
         _free = 0;
         return 0; 
      }
      _bitmap[obj]._bit = false;

      return (void *) &_arena[obj*_objectSize];
}

void Arena::deallocate ( void *object )
{
   int obj = ((char *) object - _arena)/ _objectSize;

   _free = 1;
   _bitmap[obj]._bit = true;
}


inline void * Allocator::allocate ( size_t size )
{

   size_t realSize = NANOS_ALIGNED_MEMORY_OFFSET (0, size, 16);
   size_t headerSize = NANOS_ALIGNED_MEMORY_OFFSET(0,sizeof(ObjectHeader),16);
   size_t allocSize = realSize + headerSize;

   ObjectHeader * ptr;
   unsigned int i;

   for ( i = 0; i < _nArenas ; i++ )
   {
      if ( _arenas[i]->getObjectSize() == allocSize ) break;
   }

   if ( i == _nArenas ) {
      // no arena found for that size, create a new one
      ObjectArena *arena = new ObjectArena(allocSize);

      _arenas.push_back(arena);
      _nArenas++;

   }

   Arena *arena = _arenas[i]; 
   while ( ptr == NULL ) {
      ptr = (ObjectHeader *) arena->allocate();
      if ( ptr == NULL ) { 
          if ( !arena->_next ) {
             arena->_next = new Arena(allocSize);
          }
          arena = arena->_next 
      }
   }

   ptr->_arena = arena;

   return  ((char *) ptr ) + headerSize;
}

static void deallocate ( void *object )
{
   size_t headerSize = NANOS_ALIGNED_MEMORY_OFFSET(0,sizeof(ObjectHeader),16);
   ObjectHeader * ptr = (ObjectHeader *) ( ((char *)object) - headerSize );
   Arena *arena = ptr->_arena;
   arena->deallocate(ptr);
}

#endif
