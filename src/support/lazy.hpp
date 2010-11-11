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

#ifndef _NANOS_LAZY_INIT
#define _NANOS_LAZY_INIT

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

template <class T>
class LazyInit {
   private:
      bool _init;
      char _storage[sizeof(T)] __attribute__((aligned(8)));

      void construct ()
      {
         _init = true;
         new (&_storage) T();
      }

      void destroy ()
      {
         T * ptr = (T *) &_storage;
         ptr->~T();
      }

   public:
      LazyInit() : _init(false) {}

      ~LazyInit () { if (_init) destroy();  }

      T * operator-> ()
      {
         if (unlikely(!_init)) construct();
         return (T *)&_storage;
      }

      T & operator* () {  return *(operator->());   }

      bool isInitialized() { return _init; }
};

#endif
