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

#ifndef _NANOS_LIB_WDDEQUE_DECL
#define _NANOS_LIB_WDDEQUE_DECL

#include <list>
#include "atomic.hpp"
#include "debug.hpp"
#include "workdescriptor.hpp"
#include "basethread.hpp"

namespace nanos
{

   class SchedulePredicate
   {

      public:
         SchedulePredicate () {}
         virtual bool operator() ( WorkDescriptor *wd ) = 0;
         virtual ~SchedulePredicate() {}
   };


   class WDDeque
   {

      private:
         typedef std::list<WorkDescriptor *> BaseContainer;

         BaseContainer     _dq;
         Lock              _lock;

         WDDeque ( const WDDeque & );
         const WDDeque & operator= ( const WDDeque & );

      public:
         WDDeque() : _dq(), _lock() {}
         ~WDDeque() {}

         bool empty ( void ) const;

         void push_front ( WorkDescriptor *wd );
         void push_back( WorkDescriptor *wd );
         WorkDescriptor * pop_front ( BaseThread *thread );
         WorkDescriptor * pop_front ( BaseThread *thread, SchedulePredicate &predicate );
         WorkDescriptor * pop_back ( BaseThread *thread );
         WorkDescriptor * pop_back ( BaseThread *thread, SchedulePredicate &predicate );

         bool removeWD( BaseThread *thread, WorkDescriptor * toRem );
   };

}

#endif

