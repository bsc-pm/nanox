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

#ifndef _NANOS_ATOMIC_DECL
#define _NANOS_ATOMIC_DECL


/* TODO: move to configure
#include <ext/atomicity.h>
#ifndef _GLIBCXX_ATOMIC_BUILTINS
#error "Atomic gcc builtins support is mandatory at this point"
#endif
*/

namespace nanos {

   template<typename T>
   class Atomic
   {

      private:
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
         T     _value;
#else
         volatile T     _value;
#endif

      public:
         // constructor
         Atomic () : _value() {}

         Atomic ( T init ) : _value( init ) {}

         // copy constructor
         Atomic ( const Atomic &atm ) : _value( atm._value ) {}

         // assignment operator
         Atomic & operator= ( const Atomic &atm );
         Atomic & operator= ( const T val );
         // destructor
         ~Atomic() {}

         T fetchAndAdd ( const T& val=1 );
         T addAndFetch ( const T& val=1 );
         T fetchAndSub ( const T& val=1 );
         T subAndFetch ( const T& val=1 );
         T value() const;

         operator T() const { return value(); }

         //! pre-increment ++
         T operator++ ();
         T operator-- ();

         //! post-increment ++
         T operator++ ( int val );
         T operator-- ( int val );

         //! += operator
         T operator+= ( const T val );
         T operator+= ( const Atomic<T> &val );

         T operator-= ( const T val );
         T operator-= ( const Atomic<T> &val );

#if 0 // Deprecated
         //! equal operator
         bool operator== ( const Atomic<T> &val );
         bool operator!= ( const Atomic<T> &val );

         bool operator< (const Atomic<T> &val );
         bool operator> ( const Atomic<T> &val ) const;
         bool operator<= ( const Atomic<T> &val );
         bool operator>= ( const Atomic<T> &val );
#endif

         // other atomic operations

         //! compare and swap
         bool cswap ( const Atomic<T> &oldval, const Atomic<T> &newval );

#ifdef HAVE_NEW_GCC_ATOMIC_OPS
         // note that the caller becomes responsible of accessing the shared
         // storage in a non-racy way
         T& override ();
#else
         volatile T & override ();
#endif
   };

   void memoryFence ();

   template<typename T>
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   bool compareAndSwap( T *ptr, T oldval, T  newval );
#else
   bool compareAndSwap( volatile T *ptr, T oldval, T  newval );
#endif

} // namespace nanos

#endif
