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

#ifndef SMARTPOINTER
#define SMARTPOINTER

#include "debug.hpp"

namespace nanos {

struct default_deleter {
   public:
      template < typename T >
      void operator()( T *ptr ) {
         delete ptr;
      };
};

template < typename T, class deleter = default_deleter >
class shared_pointer {
   private:
      T *pointer;
      deleter delete_function;
   public:
      shared_pointer() : pointer(NULL), delete_function() {}

      shared_pointer( T* ptr ) : pointer(ptr), delete_function()
      {
         if( ptr )
            ptr->reference();
      }

      shared_pointer( shared_pointer const& other ) :
         pointer(other.pointer)
      {
         if( pointer )
            pointer->reference();
      }

#ifdef HAVE_CXX11
      shared_pointer( shared_pointer &&other ) :
         pointer( other.pointer )
      {
         // Other will be deleted soon.
         // No need to reference/unreference, just
         // invalidate other.pointer
         other.pointer = 0;
      }
#endif

      virtual ~shared_pointer()
      {
         if( pointer && pointer->unreference() == 0 ) 
            delete_function( pointer );
      }

      shared_pointer const& operator=( shared_pointer const& other ) {
         if( pointer && pointer->unreference() == 0 ) {
            delete_function( pointer );
         }
         pointer = other.pointer;
         if( pointer )
            pointer->reference();
         return *this;
      }

#ifdef HAVE_CXX11
      shared_pointer& operator=( shared_pointer &&other ) {
         // Perform a cleanup that takes care of at least those parts of the
         // destructor that have side effects. Be sure to leave the object
         // in a destructible and assignable state.
         if( pointer && pointer->unreference() == 0 )
            delete_function( pointer );

         // No need to reference/unreference "other"'s pointee object,
         //  since it is a temporary instance
         pointer = other.pointer;
         other.pointer = 0;
         return *this;
      }
#endif

      void reset( T *ptr )
      {
         if( ptr )
            ptr->reference();

         if( pointer && pointer->unreference() == 0 ) 
            delete_function( pointer );
         
         pointer = ptr;
      }

      T *get() { return pointer; }

      const T *get() const { return pointer; }

      T *operator->() { return pointer; }

      operator T*()   { return pointer; } //!< Implicit type cast pointer operator

      T &operator*()
      {
         fatal_cond0( !pointer, "Trying to dereference a null pointer." );
         if( pointer )
            return *pointer;
      }

      operator T&()   { return operator*(); } //!< Implicit type cast reference operator

      template< class other_deleter >
      void swap( shared_pointer<T,other_deleter> &other ) {
         T *tmp_ptr = other.pointer;
         other.pointer = pointer;
         pointer = tmp_ptr;
      }
};

template < typename T, class deleter = default_deleter >
class unique_pointer {
   private:
      T *pointer;
      deleter delete_function;
   public:
      unique_pointer() : pointer(NULL) {}

      unique_pointer( T* ptr ) : pointer( ptr ), delete_function()
      {
      }

      unique_pointer( unique_pointer &other ) :
         pointer(other.pointer), delete_function()
      {
         other.pointer = NULL;
      }

      virtual ~unique_pointer() { if( pointer ) delete pointer; }

      unique_pointer const& operator=( unique_pointer &other ) {
         if( pointer )
            delete_function( pointer );
         pointer = other.pointer;
         other.pointer = NULL;
         return *this;
      }

      operator bool() const { return pointer != NULL; };

      const T* get() const { return pointer; }

      T* get() { return pointer; }

      void reset( T *ptr )
      {
         if( pointer ) 
            delete_function( pointer );
         pointer = ptr;
      }

      T *operator->() { return pointer; }

      T &operator*()
      {
         fatal_cond0( !pointer, "Trying to dereference a null pointer." );
         return *pointer;
      }

      template< class other_deleter >
      void swap( shared_pointer<T,other_deleter> &other ) {
         T *tmp_ptr = other.pointer;
         other.pointer = pointer;
         pointer = tmp_ptr;
      }
};

} // namespace nanos
#endif
