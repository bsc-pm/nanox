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

#include <list>
#include <cstdlib>
#include <cstring>
#include "malign.hpp"
#include <iostream>

#define NANOS_CACHELINE 128 /* FIXME: This definition must be architectural dependant */
#define NANOS_OBJECTS_PER_ARENA 100

namespace nanos
{

/*! \class Allocator
 */
class Allocator
{
   private:
     /*! \class Arena
      */
      class Arena
      {
         private: /* Arena data members and disabled constructors */
            static const size_t numObjects = NANOS_OBJECTS_PER_ARENA;      /** Number of maximum objects allocated in this arena*/

            union bitmap_entry {
               bool             _bit;
               char             pad[NANOS_CACHELINE];
            };                                         /**< bitmap_entry struct */

            size_t            _objectSize;             /**< Object size in current Arena  */
            char *            _arena;                  /**< Memory region used by Arena */
            bitmap_entry      *_bitmap;                /**< Bit map (free/busy) */
            Arena             *_next;                  /**< Next Arena in the list */
            /*! \brief Arena copy constructor (disabled)
             */
            Arena ( const Arena &a );
            /*! \brief Arena copy assignment operator (disabled)
             */
            Arena & operator= ( const Arena &a );
           /*! \brief Arena default constructor (disabled)
            */
            Arena ();
         public: /* Arena method members */
           /*! \brief Arena constructor
            */
            Arena ( size_t objectSize ) : _objectSize(objectSize), _next (NULL)
            {
               // TODO: fusionar malloc
               _arena = (char *) malloc( objectSize * numObjects );
               _bitmap = (bitmap_entry *) malloc( sizeof(bitmap_entry) * numObjects ) ;
               for ( size_t i = 0; i < numObjects; i++ ) _bitmap[i]._bit = true;
            }
           /*! \brief Arena destructor
            */
            ~Arena ()
            {
               delete _next;
               free(_arena);
               free(_bitmap);
            }
           /*! \brief Returns the size of allocated object
            */
            size_t getObjectSize ( void ) const ; 
           /*! \brief Returns a free object address (and mark it as busy)
            */
            void * allocate ( void ) ;
           /*! \brief Mark 'object' as free
            */
            void deallocate ( void *object ) ;
           /*! \brief Returns next Arena object in the list
            */
            Arena * getNext ( void ) const;
           /*! \brief Set as next Arena object 'a'
            */
            void setNext ( Arena * a );
      };

      struct ObjectHeader { Arena *_arena; };

   private: /* Allocator data members */
      typedef std::list<Arena *>    ArenaCollection;
      ArenaCollection               _arenas;      /**< Vector of Arenas in Allocator*/
     /*! \brief Allocator copy constructor (disabled)
      */
      Allocator ( const Allocator &a );
     /*! \brief Allocator copy assignment operator (disabled)
      */
      Allocator & operator= ( const Allocator &a );

   public: /* Allocator method members */
    /*! \brief Allocator default constructor 
     */
     Allocator () : _arenas() { }
    /*! \brief Allocator destructor 
     */
     ~Allocator () { for ( ArenaCollection::iterator it = _arenas.begin(); it != _arenas.end(); it++ ) delete (*it); }
    /*! \brief Allocates 'size' bytes in memory and returns memory pointer
     *
     *  This function will check in his list of Arenas looking for one who
     *  stores objects of the given 'size'. If Allocator finds an Arena region
     *  with this 'size' will ask to it for a new object. If none of the current
     *  Arenas works with the given 'size' Allocator will create a new Arena entry
     *  in order to manage this (and future) objects of the given 'size'.
     */
     void * allocate ( size_t size ) ;
    /*! \brief Deallocates 'object' (object has a header which identifies related Arena
     */
     static void deallocate ( void *object ) ;
};

}; // namespace: nanos

#endif
