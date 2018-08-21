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

#ifndef _NANOS_LAZY_INIT
#define _NANOS_LAZY_INIT

#include "lazy_decl.hpp"
#include "atomic.hpp"
#include "lock.hpp"

namespace nanos {

template <class T>
inline void LazyInit<T>::construct ()
{
   _ptr = new (_storage) T();
}

template <class T>
inline void LazyInit<T>::destroy ()
{
    _ptr->~T();
}

template <class T>
inline LazyInit<T>::~LazyInit ()
{
   if (_ptr != NULL) destroy();
}

template <class T>
inline T * LazyInit<T>::operator-> ()
{
   // Double checked lock idiom -- Nanox is full of concurrent accesses to lazy
   // initialized data structures.
   if ( __builtin_expect(_ptr == NULL,0) )
   {
      SyncLockBlock lock(_initLock);

      if ( __builtin_expect(_ptr == NULL,0) )
         construct();
   }

   return _ptr;
}

template <class T>
inline T & LazyInit<T>::operator* ()
{
   return *(operator->());
}

template <class T>
inline bool LazyInit<T>::isInitialized()
{
   // Due to the monotonicity of the double checked lock idiom, this checks is
   // actually safe: indeed _ptr either is NULL and becomes !NULL or it will be
   // NULL for all the lifetime of this object.
   return _ptr != NULL;
}

} // namespace nanos

#endif
