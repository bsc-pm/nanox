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

#ifndef _NANOS_LIB_WDPRIORITYQUEUE_DECL_H
#define _NANOS_LIB_WDPRIORITYQUEUE_DECL_H

#include <queue>
#include "atomic_decl.hpp"
#include "debug.hpp"
#include "workdescriptor_decl.hpp"
#include "basethread_decl.hpp"
#include "wddeque_decl.hpp"

namespace nanos
{
   /*! \brief Class used to compare WDs by priority.
    *  \see WDPriorityQueue::push
    */
   struct WDPriorityComparison
   {
      bool operator() ( const WD* wd1, const WD* wd2 ) const
      {
         return wd1->getPriority() > wd2->getPriority();
      }
   };
   
   class WDPriorityQueue : public WDDeque
   {
      private:
         // TODO (gmiranda): Measure if vector is better as a container
         typedef std::list<WorkDescriptor *> BaseContainer;

         BaseContainer     _dq;
         Lock              _lock;
         size_t            _nelems;

      private:
         /*! \brief WDPriorityQueue copy constructor (private)
          */
         WDPriorityQueue ( const WDPriorityQueue & );
         /*! \brief WDPriorityQueue copy assignment operator (private)
          */
         const WDPriorityQueue & operator= ( const WDPriorityQueue & );
      public:
         /*! \brief WDPriorityQueue default constructor
          */
         WDPriorityQueue() : _dq(), _lock() {}
         /*! \brief WDPriorityQueue destructor
          */
         ~WDPriorityQueue() {}

         bool empty ( void ) const;
         size_t size() const;

         void push ( WorkDescriptor *wd );

         template <typename Constraints>
         WorkDescriptor * popWithConstraints ( BaseThread *thread );
         template <typename Constraints>
         bool removeWDWithConstraints( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

         WorkDescriptor * pop ( BaseThread *thread );

         bool removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

         void increaseTasksInQueues( int tasks );
         void decreaseTasksInQueues( int tasks );
   };

}

#endif

