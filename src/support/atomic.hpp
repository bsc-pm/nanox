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

#ifndef _NANOS_ATOMIC
#define _NANOS_ATOMIC

#include "atomic_decl.hpp"
#include "basethread_decl.hpp"
#include "compatibility.hpp"
#include "nanos-int.h"
#include <algorithm> // for min/max
#include "instrumentationmodule_decl.hpp"

/* TODO: move to configure
#include <ext/atomicity.h>
#ifndef _GLIBCXX_ATOMIC_BUILTINS
#error "Atomic gcc builtins support is mandatory at this point"
#endif
*/

namespace nanos {

template<typename T>
inline T Atomic<T>::fetchAndAdd ( const T& val )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   return __atomic_fetch_add(&_value, val, __ATOMIC_ACQ_REL);
#else
   return __sync_fetch_and_add( &_value,val );
#endif
}

template<typename T>
inline T Atomic<T>::addAndFetch ( const T& val )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   return __atomic_add_fetch(&_value, val, __ATOMIC_ACQ_REL);
#else
   return __sync_add_and_fetch( &_value,val );
#endif
}

template<typename T>
inline T Atomic<T>::fetchAndSub ( const T& val )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   return __atomic_fetch_sub( &_value, val, __ATOMIC_ACQ_REL);
#else
   return __sync_fetch_and_sub( &_value,val );
#endif
}

template<typename T>
inline T Atomic<T>::subAndFetch ( const T& val )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   return __atomic_sub_fetch( &_value, val, __ATOMIC_ACQ_REL);
#else
   return __sync_sub_and_fetch( &_value,val );
#endif
}

template<typename T>
inline T Atomic<T>::value() const
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   return __atomic_load_n(&_value, __ATOMIC_ACQUIRE);
#else
   return _value;
#endif
}

template<typename T>
inline T Atomic<T>::operator++ ()
{
   return addAndFetch();
}

template<typename T>
inline T Atomic<T>::operator-- ()
{
   return subAndFetch();
}

template<typename T>
inline T Atomic<T>::operator++ ( int val )
{
   return fetchAndAdd();
}

template<typename T>
inline T Atomic<T>::operator-- ( int val )
{
   return fetchAndSub();
}

template<typename T>
inline T Atomic<T>::operator+= ( const T val )
{
   return addAndFetch(val);
}

template<typename T>
inline T Atomic<T>::operator+= ( const Atomic<T> &val )
{
   return addAndFetch(val.value());
}

template<typename T>
inline T Atomic<T>::operator-= ( const T val )
{
   return subAndFetch(val);
}

template<typename T>
inline T Atomic<T>::operator-= ( const Atomic<T> &val )
{
   return subAndFetch(val.value());
}

#if 0 // Deprecated
template<typename T>
inline bool Atomic<T>::operator== ( const Atomic<T> &val )
{
   return value() == val.value();
}

template<typename T>
inline bool Atomic<T>::operator!= ( const Atomic<T> &val )
{
   return value() != val.value();
}

template<typename T>
inline bool Atomic<T>::operator< (const Atomic<T> &val )
{
   return value() < val.value();
}

template<typename T>
inline bool Atomic<T>::operator> ( const Atomic<T> &val ) const
{
   return value() > val.value();
}

template<typename T>
inline bool Atomic<T>::operator<= ( const Atomic<T> &val )
{
   return value() <= val.value();
}

template<typename T>
inline bool Atomic<T>::operator>= ( const Atomic<T> &val )
{
   return value() >= val.value();
}
#endif

template<typename T>
inline bool Atomic<T>::cswap ( const Atomic<T> &oldval, const Atomic<T> &newval )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   // FIXME: The atomics passed are const
   T* oldv = const_cast<T*>(&oldval._value);
   T* newv = const_cast<T*>(&newval._value);
   return __atomic_compare_exchange_n( &_value, oldv, newv,
         /* weak */ false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE );
#else
   return __sync_bool_compare_and_swap ( &_value, oldval.value(), newval.value() );
#endif
}

#ifdef HAVE_NEW_GCC_ATOMIC_OPS
template<typename T>
inline T& Atomic<T>::override()
{
   return _value;
}
#else
template<typename T>
inline volatile T& Atomic<T>::override()
{
   // Kludgy
   return _value;
}
#endif

template<typename T>
inline Atomic<T> & Atomic<T>::operator= ( const T val )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   __atomic_store_n(&_value, val, __ATOMIC_RELEASE);
#else
   _value = val;
#endif
   return *this;
}

template<typename T>
inline Atomic<T> & Atomic<T>::operator= ( const Atomic<T> &val )
{
   return operator=( val._value );
}

inline void memoryFence ()
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   __atomic_thread_fence(__ATOMIC_ACQ_REL);
#else
#ifndef __MIC__
    __sync_synchronize();
#else
    __asm__ __volatile__("" ::: "memory");
#endif
#endif
}

#ifdef HAVE_NEW_GCC_ATOMIC_OPS
template<typename T>
inline bool compareAndSwap( T *ptr, T oldval, T  newval )
{
   return __atomic_compare_exchange_n(ptr, &oldval, newval,
         /* weak */ false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE );
}
#else
template<typename T>
inline bool compareAndSwap( volatile T *ptr, T oldval, T  newval )
{
    return __sync_bool_compare_and_swap ( ptr, oldval, newval );
}
#endif

} // namespace nanos

#endif
