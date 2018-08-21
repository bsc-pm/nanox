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

#ifndef _NANOS_LAZY_INIT_DECL
#define _NANOS_LAZY_INIT_DECL

#include "atomic_decl.hpp"
#include "lock_decl.hpp"

namespace nanos {

template <class T>
class LazyInit {
   private:
      T *  _ptr;
      char _storage[sizeof(T)] __attribute__((aligned(8)));
      Lock _initLock;

      void construct ();

      void destroy ();

   public:
      LazyInit() : _ptr(NULL) {}

      ~LazyInit ();

      T * operator-> ();

      T & operator* ();

      bool isInitialized();
};


} // namespace nanos

#endif
