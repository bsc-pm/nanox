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

using namespace nanos;

inline size_t Allocator::Arena::getObjectSize () const
{
   return _objectSize;
}

void * Allocator::Arena::allocate ( void )
{
   unsigned int obj;

   for ( obj = 0 ; obj < numObjects ; obj++ ) {
      if ( _bitmap[obj]._bit ) break;
   }

   if (obj == numObjects) {
//      _free = 0;
      return NULL; 
   }

      _bitmap[obj]._bit = false;

      return (void *) &_arena[obj*_objectSize];
}

void Allocator::Arena::deallocate ( void *object )
{
   int obj = ((char *) object - _arena)/ _objectSize;

   //_free = 1;
   _bitmap[obj]._bit = true;
}

inline Allocator::Arena * Allocator::Arena::getNext ( void ) const
{
   return _next;
}

inline void Allocator::Arena::setNext ( Arena * a )
{
   _next = a;
}

inline void * Allocator::allocate ( size_t size )
{

   size_t realSize = NANOS_ALIGNED_MEMORY_OFFSET (0, size, 16);
   size_t headerSize = NANOS_ALIGNED_MEMORY_OFFSET(0,sizeof(ObjectHeader),16);
   size_t allocSize = realSize + headerSize;

   ObjectHeader * ptr = NULL;
   unsigned int i;

   for ( i = 0; i < _nArenas ; i++ ) {
      if ( _arenas[i]->getObjectSize() == allocSize ) break;
   }

   if ( i == _nArenas ) {
      // no arena found for that size, create a new one
      Arena *arena = new Arena(allocSize);

      _arenas.push_back(arena);
      _nArenas++;

   }

   Arena *arena = _arenas[i];

   while ( ptr == NULL ) {
      ptr = (ObjectHeader *) arena->allocate();
      if ( ptr == NULL ) { 
          if ( arena->getNext() == NULL ) {
             arena->setNext( new Arena(allocSize) );
          }
          arena = arena->getNext(); 
      }
   }

   ptr->_arena = arena;

   return  ((char *) ptr ) + headerSize;
}

inline void Allocator::deallocate ( void *object )
{
   size_t headerSize = NANOS_ALIGNED_MEMORY_OFFSET(0,sizeof(ObjectHeader),16);
   ObjectHeader * ptr = (ObjectHeader *) ( ((char *)object) - headerSize );
   Arena *arena = ptr->_arena;
   arena->deallocate(ptr);
}

#endif
