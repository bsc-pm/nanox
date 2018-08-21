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

#ifndef _NANOS_LIB_WDDEQUE_DECL_H
#define _NANOS_LIB_WDDEQUE_DECL_H

#include <list>
#include <functional>
#include <map>

#include "debug.hpp"
#include "atomic_decl.hpp"
#include "lock_decl.hpp"

#include "basethread_fwd.hpp"

#include "workdescriptor_decl.hpp"

#define NANOS_ABA_MASK (15)
#define NANOS_ABA_PTR(x) ((volatile WDNode *)(((uintptr_t)(x))& ~(uintptr_t)NANOS_ABA_MASK))
#define NANOS_ABA_CTR(x) (((uintptr_t)(x))&NANOS_ABA_MASK)
#define NANOS_ABA_COMPOSE(x, y) (void *)(((uintptr_t)NANOS_ABA_PTR(x)) | ((NANOS_ABA_CTR(y) + 1)&NANOS_ABA_MASK))

namespace nanos {

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
         virtual ~WDPool() {}

         virtual bool empty ( void ) const = 0;
         virtual size_t size() const = 0; /*FIXME: Try to remove this functions, use empty, there is a global counter for ready tasks  */

         virtual void push_front ( WorkDescriptor *wd ) = 0;
         virtual void push_back( WorkDescriptor *wd ) = 0;
         virtual WorkDescriptor * pop_front ( BaseThread *thread ) = 0;
         virtual WorkDescriptor * pop_back ( BaseThread *thread ) = 0;

         virtual bool removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next ) = 0;

         template <typename Constraints>
         WorkDescriptor * popFrontWithConstraints ( BaseThread const *thread );
         template <typename Constraints>
         WorkDescriptor * popBackWithConstraints ( BaseThread const *thread );
         template <typename Constraints>
         bool removeWDWithConstraints( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );
         
         /*! \brief Returns the lock object, for batch operations. */
         virtual Lock& getLock() = 0;
         
         /*! \brief Pushes back WorkDescriptors, for batch operations.
          *  \note The lock must be acquired and release externally!
          */
         virtual void push_front( WD** wds, size_t numElems ) = 0;
         
         /*! \brief Inserts WDs in the front, for batch operations.
          *  \note The lock must be acquired and release externally!
          */
         virtual void push_back( WD** wds, size_t numElems ) = 0;

         /*! \brief Returns true if an element can be dequeued
          */

         virtual bool testDequeue() { return !empty(); }
   };

   class WDDeque : public WDPool
   {
      private:
         typedef std::list<WorkDescriptor *> BaseContainer;
         typedef std::map< const Device *, Atomic<unsigned int> > WDDeviceCounter;

         BaseContainer     _dq;
         Lock              _lock;
         size_t            _nelems;
         WDDeviceCounter   _ndevs;
         bool              _deviceCounter;


      private:
         /*! \brief WDDeque copy constructor (private)
          */
         WDDeque ( const WDDeque & );
         /*! \brief WDDeque copy assignment operator (private)
          */
         const WDDeque & operator= ( const WDDeque & );

         void increaseTasksInQueues( int tasks, int increment = 1 );
         void decreaseTasksInQueues( int tasks, int decrement = 1 );

         void increaseDeviceCounter ( WorkDescriptor *wd );
         void decreaseDeviceCounter ( WorkDescriptor *wd );

      public:
         /*! \brief WDDeque default constructor
          */
         WDDeque( bool enableDeviceCounter = true );
         /*! \brief WDDeque destructor
          */
         ~WDDeque() {}

         bool empty ( void ) const;
         size_t size() const;

         void push_front ( WorkDescriptor *wd );
         void push_back( WorkDescriptor *wd );
         
         Lock& getLock();
         void push_front( WD** wds, size_t numElems );
         void push_back( WD** wds, size_t numElems );

         template <typename Constraints>
         WorkDescriptor * popFrontWithConstraints ( BaseThread const *thread );
         template <typename Constraints>
         WorkDescriptor * popBackWithConstraints ( BaseThread const *thread );
         template <typename Constraints>
         bool removeWDWithConstraints( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

         WorkDescriptor * pop_front ( BaseThread *thread );
         WorkDescriptor * pop_back ( BaseThread *thread );

         bool removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

         bool testDequeue();

         void transferElemsFrom( WDDeque &dq );
         template <typename Test>
         void iterate ();
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
            _head = _tail = NEW WDNode(NULL,NULL);
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

         void push_front( WD** wds, size_t numElems );
         void push_back( WD** wds, size_t numElems );
         WorkDescriptor * pop_front ( BaseThread *thread );
         WorkDescriptor * pop_back ( BaseThread *thread );

         bool removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

   };
   
   /*! \brief Class used to compare WDs by priority.
    *  \see WDPriorityQueue::push
    */
   template <typename T>
   struct WDPriorityComparisonBase : public std::binary_function< WD*, WD*, bool>
   {
      /*! \brief Type of the functor */
      typedef std::const_mem_fun_t<T, WD> PriorityValueFun;
      
      PriorityValueFun _getter;
      
      WDPriorityComparisonBase( PriorityValueFun functor ) : _getter( functor ) {}
   };
   
   /*! \brief Class used to compare WDs by priority.
    */
   template <typename T>
   struct WDPriorityComparison : public WDPriorityComparisonBase<T>
   {
      typedef WDPriorityComparisonBase<T> Base;
      
      WDPriorityComparison( typename Base::PriorityValueFun functor ) 
         : Base( functor ) {}
      
      bool operator() ( const WD *wd1, const WD *wd2 ) const
      {
         return this->_getter( wd1 ) > this->_getter( wd2 );
      }
   };
   
   /*! \brief Class used to compare WDs by priority reversely.
    */
   template <typename T>
   struct WDPriorityComparisonReverse : public WDPriorityComparisonBase<T>
   {
      typedef WDPriorityComparisonBase<T> Base;
      
      WDPriorityComparisonReverse( typename Base::PriorityValueFun functor ) 
         : Base( functor ) {}
      bool operator() ( const WD *wd1, const WD *wd2 ) const
      {
         return this->_getter( wd1 ) <= this->_getter( wd2 );
      }
   };

   /*! \brief Namespace used to refer WDPriorityQueue BaseContainer.
    */
   namespace WDPQ
   {
       typedef std::list<WorkDescriptor *> BaseContainer;
   }

   template<typename T = WD::PriorityType>
   class WDPriorityQueue : public WDPool
   {
      public:
         typedef T         type;
         typedef std::const_mem_fun_t<T, WD> PriorityValueFun;
         typedef std::map< const Device *, Atomic<unsigned int> > WDDeviceCounter;

      private:
         // TODO (gmiranda): Measure if vector is better as a container
         WDPQ::BaseContainer _dq;
         Lock                _lock;
         size_t              _nelems;
         /*! \brief When this is enabled, elements with the same priority
          * as the one in the back will be inserted at the back.
          * \note When this is enabled, it will override the LIFO behaviour
          * in the above case.
          */
         bool              _optimise;
         
         /*! \brief Revert insertion */
         bool              _reverse;

         /*! \brief Counts the number of WDs in the queue for each architecture */
         WDDeviceCounter   _ndevs;
         bool              _deviceCounter;

         /*! \brief Functor that will be used to get the priority or
          *  deadline */
         PriorityValueFun  _getter;
         
         /*! \brief Max and min priorities found at the queue. */
         WD::PriorityType  _maxPriority, _minPriority;
      

      private:
         /*! \brief WDPriorityQueue copy constructor (private)
          */
         WDPriorityQueue ( const WDPriorityQueue & );
         /*! \brief WDPriorityQueue copy assignment operator (private)
          */
         const WDPriorityQueue & operator= ( const WDPriorityQueue & );
         
         /*! \brief Inserts a WD in the expected order.
          *  \param fifo Insert WDs with the same after the current ones?
          */
         void insertOrdered ( WorkDescriptor *wd, bool fifo = true );
         void insertOrdered ( WorkDescriptor ** wds, size_t numElems, bool fifo = true );
         
         /*! \brief Performs upper bound reversely or not depending on the settings */
         WDPQ::BaseContainer::iterator upper_bound( const WD *wd );
        
         /*! \brief Performs lower bound reversely or not depending on the settings */
         WDPQ::BaseContainer::iterator lower_bound( const WD *wd );


         void increaseTasksInQueues( int tasks, int increment = 1 );
         void decreaseTasksInQueues( int tasks, int decrement = 1 );

         void increaseDeviceCounter ( WorkDescriptor *wd );
         void decreaseDeviceCounter ( WorkDescriptor *wd );

      public:
         /*! \brief WDPriorityQueue default constructor
          */
         WDPriorityQueue( bool enableDeviceCounter = true, bool optimise = true, bool reverse = false,
               PriorityValueFun getter = std::mem_fun( &WD::getPriority ) );
         
         /*! \brief WDPriorityQueue destructor
          */
         ~WDPriorityQueue() {}

         bool empty ( void ) const;
         size_t size() const;

         void push_back( WorkDescriptor *wd );
         void push_front( WorkDescriptor *wd );
         
         Lock& getLock();
         void push_front( WD** wds, size_t numElems );
         void push_back( WD** wds, size_t numElems );

         template <typename Constraints>
         WorkDescriptor * popFrontWithConstraints ( BaseThread *thread );
         template <typename Constraints>
         WorkDescriptor * popBackWithConstraints ( BaseThread *thread );
         template <typename Constraints>
         bool removeWDWithConstraints( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

         WorkDescriptor * pop_back ( BaseThread *thread );
         WorkDescriptor * pop_front ( BaseThread *thread );


         bool removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );

         /*! \brief Reorders the list a WD in the current queue.
          * It is needed when the priority of a WD is changed.
          * \note This just removes and pushes the WD into the list.
          * \return If the WD was found or not.
          * \note This method sets the lock upon entry (using LockBlock).
          */
         bool reorderWD( WorkDescriptor *wd );
         
         /*! \brief Returns the highest priority, without blocking.
          */
         WD::PriorityType maxPriority() const;
         
         /*! \brief Returns the lowest priority, without blocking.
          */
         WD::PriorityType minPriority() const;

         bool testDequeue();
   };


} // namespace nanos

#endif

