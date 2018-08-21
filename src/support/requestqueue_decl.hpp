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

#ifndef _REQUESTQUEUE_DECL
#define _REQUESTQUEUE_DECL
#include <list>
#include <map>
#include "atomic_decl.hpp"
#include "lock_decl.hpp"

namespace nanos {

template <class T>
class RequestQueue {
   std::list< T * > _queue;
   Lock _lock;
   public:
   RequestQueue();
   ~RequestQueue();
   void add( T * elem );
   T *fetch();
   T *tryFetch();
};

template <class T>
class RequestMap {
   std::map< uint64_t, T * > _map;
   Lock _lock;
   public:
   RequestMap();
   ~RequestMap();
   void add( uint64_t key, T * elem );
   T *fetch( uint64_t key );
   T *tryFetch( uint64_t key );
};

} // namespace nanos

#endif /* _REQUESTQUEUE_DECL */
