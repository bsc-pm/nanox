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
#include "debug.hpp"
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
   unsigned long offset = ((char *) object - _arena);
   unsigned long index = offset / _objectSize;

   ensure0 ( (offset % _objectSize) == 0, "Invalid pointer in Allocator::deallocate()");
   ensure0 ( index >= numObjects, "Invalid index in Allocator::deallocate()");

   //_free = 1;
   _bitmap[index]._bit = true;
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

   ArenaCollection::iterator it;
   for ( it = _arenas.begin(); it != _arenas.end(); it++ ) {
      if ( (*it)->getObjectSize() == allocSize ) break;
   }

   Arena *arena = NULL;

   if ( it == _arenas.end() ) {
      // no arena found for that size, create a new one
      arena = (Arena *) malloc ( sizeof(Arena) );
      new ( arena ) Arena( allocSize );
      _arenas.push_back( arena );
   }
   else arena = *it;


   ObjectHeader * ptr = NULL;

   while ( ptr == NULL ) {
      ptr = (ObjectHeader *) arena->allocate();
      if ( ptr == NULL ) { 
          if ( arena->getNext() == NULL ) {
             Arena *next = ( Arena *) malloc ( sizeof(Arena) );
             new (next) Arena(allocSize);
             arena->setNext( next );
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
