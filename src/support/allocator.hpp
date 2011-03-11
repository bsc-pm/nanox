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
#include "allocator_decl.hpp"
#include "malign.hpp"
#include "debug.hpp"
#include "basethread.hpp"
#include <vector>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace nanos {

extern Allocator nanos_alloc;

inline Allocator & getAllocator ( void )
{
   BaseThread *my_thread = getMyThreadSafe();
   if ( my_thread == NULL ) return nanos_alloc;
   else if ( my_thread->getId() == 0 ) return nanos_alloc;
   else return my_thread->getAllocator();
}

inline size_t Allocator::Arena::getObjectSize () const
{
   return _objectSize;
}

inline void * Allocator::Arena::allocate ( void )
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

inline void Allocator::Arena::deallocate ( void *object )
{
   unsigned long offset = ((char *) object - _arena);
   unsigned long index = offset / _objectSize;

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

inline void * Allocator::allocate ( size_t size, const char* file, int line )
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

#ifdef NANOS_MEMTRACKER
   ptr->_allocator = this;

   _lock.acquire();
   memoryFence();

   if ( ptr ) {
      _blocks[ptr] = BlockInfo(allocSize, file, line);
      _localMem += allocSize;
      _localBlocks++;
      _maxLocalMem = std::max( _maxLocalMem, _localMem );
      _stats[size]._current++;
      _stats[size]._total++;
      _stats[size]._max = std::max( _stats[size]._max, _stats[size]._current );
   } else {
      throw std::bad_alloc();
   }

   memoryFence();
   _lock.release();
#endif

   return  ((char *) ptr ) + headerSize;
}

inline void Allocator::deallocate ( void *object, const char *file, int line )
{
   size_t headerSize = NANOS_ALIGNED_MEMORY_OFFSET(0,sizeof(ObjectHeader),16);
   ObjectHeader * ptr = (ObjectHeader *) ( ((char *)object) - headerSize );
   Arena *arena = ptr->_arena;

#ifdef NANOS_MEMTRACKER
   Allocator *allocator = ptr->_allocator;

   _lock.acquire();
   memoryFence();

   if ( _active == false ) {
      arena->deallocate(ptr);
      memoryFence();
      _lock.release();
      return;
   }

   AddrMap::iterator it = allocator->_blocks.find( ptr );

   if ( it != allocator->_blocks.end() ) {
      allocator->_localBlocks--;
      allocator->_localMem -= it->second._size;
#endif

      arena->deallocate(ptr);

#ifdef NANOS_MEMTRACKER
      allocator->_blocks.erase( it );
      allocator->_stats[it->second._size]._current--;
   } else {
      if ( file != NULL ) {
         message0("Trying to free invalid pointer " << ptr << " at " << file << ":" << line);
      } else {
         message0("Trying to free invalid pointer " << ptr );
      }    
   }
   memoryFence();
   _lock.release();
#endif

}

#ifdef NANOS_MEMTRACKER
inline void Allocator::showStatistics( void ) const {
   std::cout << "Showing statitstics for Allocator" << std::endl;
	message0("======================= General Memory stats ============");
        if ( _id == -1 ) {
           std::cout << "Arena id                   = " << "global" << std::endl;
        } else {
           std::cout << "Arena id                   = " << _id << std::endl;
        }
	std::cout
	    << "# of local blocks          = " << _localBlocks << std::endl
	    << "Total unfreed local memory = " << _localMem << " bytes" << std::endl
	    << "Max allocated local memory = " << _maxLocalMem << " bytes" << std::endl
	    ;
        message0("=========================================================");
	message0("======================= Unfreed blocks ==================");
	for ( AddrMap::const_iterator it = _blocks.begin(); it != _blocks.end(); it++ )
	{
	    BlockInfo const &info = it->second;
	    if ( info._file != NULL ) {
	      message0(info._size << " bytes allocated in " << info._file << ":" << info._line);
	    } else {
	      message0(info._size << " bytes allocated in an unknown location");
	    }
	}
        message0("=========================================================");
#if 0        
	message0("======================= Block Sizes Stats ===============");
	message0("Size   Unfreed   Max   Total");
	for ( SizeMap::iterator it = _stats.begin(); it != _stats.end(); it ++ ) {
	    DistrInfo &info = it->second;
	    message0(it->first << " " << info._current << " " << info._max << " " << info._total );    
	}
	message0("=========================================================");
#endif        
}
#endif

} // namespace nanos

#endif
