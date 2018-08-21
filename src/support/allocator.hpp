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

#ifndef _NANOS_ALLOCATOR_HPP
#define _NANOS_ALLOCATOR_HPP
#include "allocator_decl.hpp"
#include <vector>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace nanos {

extern Allocator *allocator;

inline size_t Allocator::Arena::getObjectSize () const
{
   return _objectSize;
}

inline void Allocator::Arena::deallocate ( void *object )
{
   unsigned long offset = ((char *) object - _arena);
   unsigned long index = offset / _objectSize;

   //if (sys.getNetwork()->getNodeNum() == 0 && _objectSize > 32768) std::cerr << "Free object size " << _objectSize << " : " << (void *) _objectSize << " addr range is " << (void *) object << ":" << ((void *) (((char*)object)+_objectSize)) << " this is arena " << (void*) this << std::endl;
   _free = true;
   _bitmap[index]._bit = true;

   //if (sys.getNetwork()->getNodeNum() == 0 && _objectSize > 32768) std::cerr << "memset from " << (void *) &_arena[index*_objectSize] << " to " << (void*) (&_arena[index*_objectSize+_objectSize]) << std::endl;
   //memset( &_arena[index*_objectSize], 0, _objectSize );
}

inline Allocator::Arena * Allocator::Arena::getNext ( void ) const
{
   return _next;
}

inline void Allocator::Arena::setNext ( Arena * a )
{
   _next = a;
}

inline void * Allocator::allocateBigObject ( size_t size )
{
   ObjectHeader * ptr = NULL;

   ptr = (ObjectHeader *) malloc( size + _headerSize );
   if ( ptr == NULL ) throw(NANOS_ENOMEM);
   ptr->_arena = NULL; 

   return  ((char *) ptr ) + _headerSize;
}

inline void * Allocator::allocate ( size_t size, const char* file, int line )
{
   if ( size > _sizeOfBig ) return allocateBigObject(size);

   /* realSize is (size + header )'s next power of 2 */
   size_t realSize = size + _headerSize + 1;
   realSize |= realSize >> 1;
   realSize |= realSize >> 2;
   realSize |= realSize >> 4;
   realSize |= realSize >> 8;
   realSize |= realSize >> 16;
   realSize++;
 

   ArenaCollection::iterator it;
   for ( it = _arenas.begin(); it != _arenas.end(); it++ ) {
      if ( (*it)->getObjectSize() == realSize ) break;
   }

   Arena *arena = NULL;

   if ( it == _arenas.end() ) {
      // no arena found for that size, create a new one
      arena = (Arena *) malloc ( sizeof(Arena) );
      if ( arena == NULL ) throw(NANOS_ENOMEM);
      new ( arena ) Arena( realSize );
      _arenas.push_back( arena );
   }
   else arena = *it;


   ObjectHeader * ptr = NULL;

   while ( ptr == NULL ) {
      ptr = (ObjectHeader *) arena->allocate();
      if ( ptr == NULL ) { 
          if ( arena->getNext() == NULL ) {
             Arena *next = ( Arena *) malloc ( sizeof(Arena) );
             if ( next == NULL ) throw(NANOS_ENOMEM);
             new (next) Arena(realSize);
             arena->setNext( next );
          }
          arena = arena->getNext(); 
      }
   }

   ptr->_arena = arena;
   //if (sys.getNetwork()->getNodeNum() == 0 && realSize > 32768) std::cerr << "Allocate object size " << realSize << " : " << (void *) realSize << " addr range is " << (void *) ptr << ":" << ((void *) (((char*)ptr)+realSize)) << " this is arena " << (void *) arena << std::endl;

   return  ((char *) ptr ) + _headerSize;
}

inline void Allocator::deallocate ( void *object, const char *file, int line )
{
   if ( object == NULL ) return;

   ObjectHeader * ptr = (ObjectHeader *) ( ((char *)object) - _headerSize );

   Arena *arena = ptr->_arena;

   // If there is no arena then it was a big object that just needs to be freed
   if ( arena == NULL )
     free(ptr);
   else
     arena->deallocate(ptr);
}

inline size_t Allocator::getObjectSize ( void *object )
{
   ObjectHeader * ptr = (ObjectHeader *) ( ((char *)object) - _headerSize );
   Arena *arena = ptr->_arena;
   return arena->getObjectSize() - _headerSize ;
}

} // namespace nanos

#endif
