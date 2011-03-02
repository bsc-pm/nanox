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
#ifndef _NANOS_ALLOCATOR_DECL
#define _NANOS_ALLOCATOR_DECL

#include <vector>
#include <cstdlib>
#include <cstring>
#include "malign.hpp"
#include <iostream>

#define CACHELINE 128 /* FIXME: This definition must be architectural dependant */

namespace nanos
{

class Allocator
{
   private:
      class Arena
      {
         private:
            static const size_t numObjects = 100;

            union bitmap_entry {
               bool             _bit;
               char             pad[CACHELINE];
            };

            size_t            _objectSize;
            char *            _arena;
            bitmap_entry      *_bitmap;
            Arena             *_next;

         public:
            Arena ( size_t objectSize ) : _objectSize(objectSize), _next (NULL)
            {
               _arena = (char *) malloc( objectSize * numObjects );
               _bitmap = (bitmap_entry *) malloc( sizeof(bitmap_entry) * numObjects ) ;
               for ( size_t i = 0; i < numObjects; i++ ) _bitmap[i]._bit = true;
            }

            ~Arena ()
            {
               if ( _next ) delete _next;
               free(_arena);
               free(_bitmap);
            }

            size_t getObjectSize ( void ) const ; 
            void * allocate ( void ) ;
            void deallocate ( void *object ) ;
            Arena * getNext ( void ) const;
            void setNext ( Arena * a );

      };

      struct ObjectHeader { Arena *_arena; };

   private: /* Allocator Data Members */
      std::vector<Arena *> _arenas;
      size_t               _nArenas;

   public:

     Allocator () : _arenas(), _nArenas(0)
     {
        _arenas.reserve(10);
     }

     void * allocate ( size_t size ) ;

     static void deallocate ( void *object ) ;
};

}; // namespace: nanos

#endif
