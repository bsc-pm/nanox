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

#ifndef _REQUESTQUEUE
#define _REQUESTQUEUE

#include "requestqueue_decl.hpp"
#include "atomic.hpp"
#include "lock.hpp"

namespace nanos {

template <class T>
RequestQueue<T>::RequestQueue() : _queue(), _lock() {
}

template <class T>
RequestQueue<T>::~RequestQueue() {
}

template <class T>
void RequestQueue<T>::add( T *elem ) {
   _lock.acquire();
   _queue.push_back( elem );
   _lock.release();
}

template <class T>
T *RequestQueue<T>::fetch() {
   T *elem = NULL;
   _lock.acquire();
   //if ( !_queue.empty() ) {
   //   for ( std::list<T *>::iterator it = _delayedPutReqs.begin(); putReqsIt != _delayedPutReqs.end(); putReqsIt++ ) {
   //      if ( (*putReqsIt)->origAddr == destAddr ) {
   //         _putReqsLock.acquire();
   //         _putReqs.push_back( *putReqsIt );
   //         _putReqsLock.release();
   //      }
   //   }
   //}
   _lock.release();
}

template <class T>
T *RequestQueue<T>::tryFetch() {
   T *elem = NULL;
   if ( _lock.tryAcquire() ) {
      if ( !_queue.empty() ) {
         elem = _queue.front();
         _queue.pop_front();
      }
      _lock.release();
   }
   return elem;
}

template <class T>
RequestMap<T>::RequestMap() : _map(), _lock() {
}

template <class T>
RequestMap<T>::~RequestMap() {
}
template <class T>
void RequestMap<T>::add( uint64_t key, T *elem ) {
   _lock.acquire();
   typename std::map< uint64_t, T * >::iterator it = _map.lower_bound( key );
   if ( it != _map.end() || _map.key_comp()( key, it->first ) ) {
      _map.insert( it, std::map< uint64_t, T * >::value_type( key, elem ) );
   } else { 
      std::cerr << "Error, key already exists." << std::endl;
   }
   _lock.release();
}

template <class T>
T *RequestMap<T>::fetch( uint64_t key ) {
   T *elem = NULL;
   _lock.acquire();
   typename std::map< uint64_t, T * >::iterator it = _map.lower_bound( key );
   if ( it != _map.end() || _map.key_comp()( key, it->first ) ) {
      std::cerr << "Error, key not found." << std::endl;
   } else {
      elem = it->second;
   }
   _lock.release();
}

} // namespace nanos

#endif /* _REQUESTQUEUE */
