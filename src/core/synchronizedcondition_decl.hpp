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

#ifndef _NANOS_SYNCHRONIZED_CONDITION_DECL
#define _NANOS_SYNCHRONIZED_CONDITION_DECL

#include <stdlib.h>
#include <list>
#include <vector>
#include "atomic.hpp"
#include "debug.hpp"
#include "workdescriptor_fwd.hpp"

namespace nanos
{
  /* \brief Represents an object that checks a given condition.
   */
   class ConditionChecker
   {
      public:
        /* \brief Constructor
         */
         ConditionChecker() {}

        /* \brief Copy constructor
         * \param cc Another ConditionChecker
         */
         ConditionChecker ( const ConditionChecker & cc ) 
         {
         }

        /* \brief Assign operator
         * \param cc Another ConditionChecker
         */
         ConditionChecker& operator=( const ConditionChecker & cc )
         {
            return *this;
         }

        /* \brief virtual destructor
         */
         virtual ~ConditionChecker() {}

        /* \brief interface used by the SynchronizedCondition to check the condition.
         */
         virtual bool checkCondition() = 0;
   };

  /* \brief Checks a templated variable for equality with a given condition.
   */
   template<typename T>
   class EqualConditionChecker : public ConditionChecker
   {
      protected:
         /**< variable wich value has to be checked. */
         volatile T*     _var;
         /**< variable value to ckeck for. */
         T               _condition;

      public:
        /* \brief Constructor
         */
         EqualConditionChecker() : ConditionChecker(), _var(NULL), _condition() {}

        /* \brief Constructor
         */
         EqualConditionChecker(volatile T* var, T condition) : ConditionChecker(), _var(var), _condition(condition) {}

        /* \brief Copy constructor
         * \param cc Another EqualConditionChecker
         */
         EqualConditionChecker ( const EqualConditionChecker & cc ) 
         {
            this->_var = cc._var;
            this->_condition = cc._condition;
         }

        /* \brief Assign operator
         * \param cc Another EqualConditionChecker
         */
         EqualConditionChecker& operator=( const EqualConditionChecker & cc )
         {
            this->_var = cc._var;
            this->_condition = cc._condition;
            return *this;
         }

        /* \brief virtual destructor
         */
         virtual ~EqualConditionChecker() {}

        /* \brief Checks the variable against the condition.
         */
         virtual bool checkCondition() {
            return ( *(this->_var) == (this->_condition) );
         }
   };

  /* \brief Checks a templated variable for being less or equal than a condition.
   */
   template<typename T>
   class LessOrEqualConditionChecker : public ConditionChecker
   {
      protected:
         /**< variable wich value has to be checked. */
         volatile T*     _var;
         /**< variable value to ckeck for. */
         T               _condition;

      public:
        /* \brief Constructor
         */
         LessOrEqualConditionChecker() : ConditionChecker(), _var(NULL), _condition() {}

        /* \brief Constructor
         */
         LessOrEqualConditionChecker(volatile T* var, T condition) : ConditionChecker(), _var(var), _condition(condition) {}

        /* \brief Copy constructor
         * \param cc Another LessOrEqualConditionChecker
         */
         LessOrEqualConditionChecker ( const LessOrEqualConditionChecker & cc ) 
         {
            this->_var = cc._var;
            this->_condition = cc._condition;
         }

        /* \brief Assign operator
         * \param cc Another LessOrEqualConditionChecker
         */
         LessOrEqualConditionChecker& operator=( const LessOrEqualConditionChecker & cc )
         {
            this->_var = cc._var;
            this->_condition = cc._condition;
            return *this;
         }

        /* \brief virtual destructor
         */
         virtual ~LessOrEqualConditionChecker() {}

        /* \brief Checks the variable against the condition.
         */
         virtual bool checkCondition() {
            return ( *(this->_var) <= (this->_condition) );
         }
   };

  /*! \brief Abstract synchronization class.
   */
   class GenericSyncCond
   {
      private:
         /**< Lock to block and unblock WorkDescriptors securely. */
         Lock _lock;

      public:
        GenericSyncCond() : _lock() {}

        virtual ~GenericSyncCond() {}

        virtual void wait() = 0;
        virtual void signal() = 0;
        virtual bool check() = 0;

        /** \brief Sets a waiter Workdescriptor.
         */
         virtual void addWaiter( WorkDescriptor* wd) = 0;

        /** \brief Returns true if there's any waiter on the condition.
         */
         virtual bool hasWaiters() = 0;

        /** \brief Returns a waiter adn removes it from the condition.
         */
         virtual WorkDescriptor* getAndRemoveWaiter() = 0;

        
        /** \brief acquire the lock.
         */
         void lock()
         {
            _lock.acquire();
            memoryFence();
         }

         /** \brief Release the lock. The wait() method can switch context
          * so it is necessary this function to be public so that the switchHelper
          * can unlock it after removing the current WD from the stack.
          */
         void unlock()
         {
            memoryFence();
            _lock.release();
         }
   };

  /*! \brief Abstract template synchronization class.
   */
   template<class _T>
   class SynchronizedCondition : GenericSyncCond
   {
      protected:
         /**< ConditionChecker associated to the SynchronizedCondition. */
         _T _conditionChecker;
      
      public:
        /* \brief Constructor
         */
         SynchronizedCondition ( ) : GenericSyncCond(), _conditionChecker() { }
         SynchronizedCondition ( _T cc ) : GenericSyncCond(), _conditionChecker(cc) { }

        /* \brief Copy constructor
         * \param sc Another SyncrhonizedCondition
         */
         SynchronizedCondition ( const SynchronizedCondition & sc ) : GenericSyncCond(), _conditionChecker( sc._conditionChecker )
         {
         }

        /* \brief Assign operator
         * \param sc Another SyncrhonizedCondition
         */
         SynchronizedCondition& operator=( const SynchronizedCondition & sc )
         {
            this._conditionChecker = sc._ConditionChecker;
            return *this;
         }

        /* \brief virtual destructor
         */
         virtual ~SynchronizedCondition() { }

        /* \brief Wait until the condition has been satisfied
         */
         void wait();

        /* \brief Signal the waiters if the condition has been satisfied. If they
         * are blocked they will be set to ready and enqueued.
         */
         void signal();

        /* \brief Change the condition checker associated to the synchronizedConditon object.
         */
         void setConditionChecker( _T cc )
         {
            _conditionChecker = cc;
         }

         bool check ()
         {
            return _conditionChecker.checkCondition();
         }
   };

  /* \brief SynchronizedCondition specialization that checks equality fon one
   * variable and with just one waiter.
   */
   template<class _T>
   class SingleSyncCond : public SynchronizedCondition<_T>
   {
      private:
         /**< Pointer to the WD waiting for the condition. */
         WorkDescriptor* _waiter;

      public:
        /* \brief Constructor
         */
         SingleSyncCond ( ) : SynchronizedCondition<_T>( ), _waiter( NULL ) { }

        /* \brief Constructor 
         * \param var Variable which value is used for synchronization.
         * \param condition Value expected
         */
         SingleSyncCond ( _T cc ) : SynchronizedCondition<_T>( cc ), _waiter( NULL ) { }

        /* \brief Copy constructor
         * \param ssc Another SingleSyncCond
         */
         SingleSyncCond ( const SingleSyncCond & ssc ) :  SynchronizedCondition<_T>( ssc ), _waiter( NULL )
         {
         }

        /* \brief Assign operator
         * \param ssc Another SingleSyncCond
         */
         SingleSyncCond& operator=( const SingleSyncCond & ssc )
         {
            this->_conditionChecker =  ssc._conditionChecker;
            return *this;
         }

        /* \brief virtual destructor
         */
         virtual ~SingleSyncCond() { }

      protected:
        /* \brief Sets the waiter to wd.
         */
         virtual void addWaiter( WorkDescriptor* wd )
         {
            _waiter = wd;
         }

        /* \brief Returns true if there's a waiter.
         */
         virtual bool hasWaiters()
         {
            return _waiter != NULL;
         }

        /* \brief Returns the waiter and sets it to NULL.
         */
         virtual WorkDescriptor* getAndRemoveWaiter()
         {
            WorkDescriptor* result = _waiter;
            _waiter = NULL;
            return result;
         }
   };

  /* \brief SynchronizedCondition specialization that checks equality fon one
   * variable and with more than one waiter.
   */
   template <class _T>
   class MultipleSyncCond : public SynchronizedCondition<_T>
   {
      private:
         /**< type vector of workdescriptors. */
         typedef std::vector<WorkDescriptor*> WorkDescriptorList;

         /**< List of WDs that wait on the condition. */
         WorkDescriptorList _waiters;

      public:
        /* \brief Constructor
         */
         MultipleSyncCond ( size_t size) : SynchronizedCondition<_T> (), _waiters()
         {
            _waiters.reserve(size);
         }

        /* \brief Constructor
         * \param cc ConditionChecker needed for
         * \param 
         */
         MultipleSyncCond (_T cc, size_t size) : SynchronizedCondition<_T> ( cc ), _waiters()
         {
            _waiters.reserve(size);
         }

        /* \brief Copy constructor
         * \param ssc Another MultipleSyncCond
         */
         MultipleSyncCond ( const MultipleSyncCond & ssc ) :  SynchronizedCondition<_T>( ssc ), _waiters()
         {
         }

        /* \brief Assign operator
         * \param ssc Another MultipleSyncCond
         */
         MultipleSyncCond& operator=( const MultipleSyncCond & ssc )
         {
            this->_conditionChecker = ssc._conditionChecker;
            return *this;
         }

        /* Set the number of waiters expected to wait on this condition.
         */
         void resize ( size_t size )
         {
            _waiters.reserve(size);
         }

        /* \brief virtual destructor
         */
         virtual ~MultipleSyncCond() { }

      protected:
        /* \brief Sets the waiter to wd.
         */
         virtual void addWaiter( WorkDescriptor* wd )
         {
            _waiters.push_back( wd );
         }

        /* \brief Returns true if there's a waiter.
         */
         virtual bool hasWaiters()
         {
            return !( _waiters.empty() );
         }

        /* \brief Returns the waiter and sets it to NULL.
         */
         virtual WorkDescriptor* getAndRemoveWaiter()
         {
            if ( _waiters.empty() )
               return (WorkDescriptor*)16;
            WorkDescriptor* reslt =  _waiters.back();
            _waiters.pop_back();
            return reslt;
         }
   };
}

#endif
