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

#ifndef _NANOS_LIB_WDDEQUE_DECL_H
#define _NANOS_LIB_WDDEQUE_DECL_H

#include <list>
#include "atomic_decl.hpp"
#include "debug.hpp"
#include "workdescriptor_decl.hpp"
#include "basethread_decl.hpp"

#define NANOS_ABA_MASK (15)
#define NANOS_ABA_PTR(x) ((volatile WDNode *)(((uintptr_t)(x))& ~(uintptr_t)NANOS_ABA_MASK))
#define NANOS_ABA_CTR(x) (((uintptr_t)(x))&NANOS_ABA_MASK)
#define NANOS_ABA_COMPOSE(x, y) (void *)(((uintptr_t)NANOS_ABA_PTR(x)) | ((NANOS_ABA_CTR(y) + 1)&NANOS_ABA_MASK))

namespace nanos
{

   class SchedulePredicate
   {

      public:
         /*! \brief SchedulePredicate default constructor
          */
         SchedulePredicate () {}
         /*! \brief SchedulePredicate destructor
          */
         virtual ~SchedulePredicate() {}
         /*! \brief SchedulePredicate function call operator (pure virtual)
          */
         virtual bool operator() ( WorkDescriptor *wd ) = 0;
   };

   class WDPool {
      private:
         /*! \brief WDPool copy constructor (private)
          */
         WDPool ( const WDPool & );
         /*! \brief WDPool copy assignment operator (private)
          */
         const WDPool & operator= ( const WDPool & );
      public:
         /*! \brief WDPool default constructor
          */
         WDPool() {}
         /*! \brief WDPool destructor
          */
         ~WDPool() {}

         virtual bool empty ( void ) const = 0;
         virtual size_t size() const = 0; /*FIXME: Try to remove this functions, use empty, there is a global counter for ready tasks  */

         virtual void push_front ( WorkDescriptor *wd ) = 0;
         virtual void push_back( WorkDescriptor *wd ) = 0;
         virtual WorkDescriptor * pop_front ( BaseThread *thread ) = 0;
         virtual WorkDescriptor * pop_back ( BaseThread *thread ) = 0;

         virtual bool removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next ) = 0;

         template <typename Constraints>
         WorkDescriptor * popFrontWithConstraints ( BaseThread *thread );
         template <typename Constraints>
         WorkDescriptor * popBackWithConstraints ( BaseThread *thread );
         template <typename Constraints>
         bool removeWDWithConstraints( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

         static void increaseTasksInQueues( int tasks );
         static void decreaseTasksInQueues( int tasks );

   };

   class WDDeque : public WDPool
   {
      private:
         typedef std::list<WorkDescriptor *> BaseContainer;

         BaseContainer     _dq;
         Lock              _lock;
         size_t            _nelems;

      private:
         /*! \brief WDDeque copy constructor (private)
          */
         WDDeque ( const WDDeque & );
         /*! \brief WDDeque copy assignment operator (private)
          */
         const WDDeque & operator= ( const WDDeque & );
      public:
         /*! \brief WDDeque default constructor
          */
         WDDeque() : _dq(), _lock() {}
         /*! \brief WDDeque destructor
          */
         ~WDDeque() {}

         bool empty ( void ) const;
         size_t size() const;

         void push_front ( WorkDescriptor *wd );
         void push_back( WorkDescriptor *wd );

         template <typename Constraints>
         WorkDescriptor * popFrontWithConstraints ( BaseThread *thread );
         template <typename Constraints>
         WorkDescriptor * popBackWithConstraints ( BaseThread *thread );
         template <typename Constraints>
         bool removeWDWithConstraints( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

         WorkDescriptor * pop_front ( BaseThread *thread );
         WorkDescriptor * pop_back ( BaseThread *thread );

         bool removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

         void increaseTasksInQueues( int tasks );
         void decreaseTasksInQueues( int tasks );
   };

   class WDLFQueue : public WDPool
   {
      private:
         class WDNode
         {
            private:
               WorkDescriptor    *_wd;      /**< WorkDescriptor */
               volatile WDNode   *_next;    /**< Next on the Queue */
            private:
              /*! \brief WDNode copy constructor (private)
               */
               WDNode ( const WDNode &);
              /*| \brief WDNode copy assignment operator (private)
               */
               const WDNode & operator= (const WDNode &);
            public:
              /*! \brief WDNode default constructor
               */
               WDNode (WorkDescriptor *wdp, WDNode *n): _wd(wdp), _next(n) {}
              /*! \brief WDNode default destructor
               */
               ~WDNode() {}

               volatile WDNode *next(void) volatile { return NANOS_ABA_PTR(_next); }
               volatile void *next_value(void) const volatile { return _next; }
               volatile void *next_addr(void) volatile { return &_next; }
               void resetNext ( void ) { _next = NULL; }

               WorkDescriptor *wd(void) const volatile { return _wd; }
               void setWD ( WorkDescriptor *w) volatile { _wd = w; }

         }; // end: class WDNode

      private:
         volatile WDNode  *_head;    /**< Queue head */
         volatile WDNode  *_tail;    /**< Queue tail */
      private:
         /*! \brief WDLFQueue copy constructor (private)
          */
         WDLFQueue ( const WDLFQueue & );
         /*! \brief WDLFQueue copy assignment operator (private)
          */
         const WDLFQueue & operator= ( const WDLFQueue & );
      public:
         /*! \brief WDLFQueue default constructor
          */
         WDLFQueue()
         {
            _head = _tail = new WDNode(NULL,NULL);
         }
         /*! \brief WDFifo destructor
          */
         ~WDLFQueue() {
            while ( NANOS_ABA_PTR( _head ) != NANOS_ABA_PTR( _tail ) ) {
//               pop_front ( NULL );
            }
            // delete NANOS_ABA_PTR( _head ); /*FIXME*/
         }

         bool empty ( void ) const;
         size_t size() const;
         void push_front ( WorkDescriptor *wd );
         void push_back( WorkDescriptor *wd );
         void push_back_node ( WDNode *node );

         WorkDescriptor * pop_front ( BaseThread *thread );
         WorkDescriptor * pop_back ( BaseThread *thread );

         bool removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

#if 0
         template <typename Constraints>
         WorkDescriptor * popFrontWithConstraints ( BaseThread *thread );
         template <typename Constraints>
         WorkDescriptor * popBackWithConstraints ( BaseThread *thread );
         template <typename Constraints>
         bool removeWDWithConstraints( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );
#endif
   };

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

   class WDPriorityQueue : public WDPool
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
         void push_back( WorkDescriptor *wd );
         void push_front( WorkDescriptor *wd );

         template <typename Constraints>
         WorkDescriptor * popWithConstraints ( BaseThread *thread );
         template <typename Constraints>
         bool removeWDWithConstraints( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

         WorkDescriptor * pop ( BaseThread *thread );
         WorkDescriptor * pop_back ( BaseThread *thread );
         WorkDescriptor * pop_front ( BaseThread *thread );


         bool removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

         void increaseTasksInQueues( int tasks );
         void decreaseTasksInQueues( int tasks );
   };


}

#endif

