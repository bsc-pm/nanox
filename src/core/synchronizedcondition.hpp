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

#ifndef _NANOS_SYNCHRONIZED_CONDITION
#define _NANOS_SYNCHRONIZED_CONDITION

#include <stdlib.h>
#include <list>
#include <vector>
#include "atomic.hpp"
#include "debug.hpp"

namespace nanos
{
   class WorkDescriptor;

  /* \brief Represents an object that checks a given condition.
   */
   class ConditionChecker
   {
      public:
        /* \brief Constructor
         */
         ConditionChecker() {}

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
         EqualConditionChecker(volatile T* var, T condition) : ConditionChecker(), _var(var), _condition(condition) {}

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
         LessOrEqualConditionChecker(volatile T* var, T condition) : ConditionChecker(), _var(var), _condition(condition) {}

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
   class SynchronizedCondition
   {
      private:
         /**< Lock to block and unblock WorkDescriptors securely. */
         Lock _lock;

         /**< ConditionChecker associated to the SynchronizedCondition. */
         ConditionChecker* _conditionChecker;

         // Disable copy constructor and assign operator.
         SynchronizedCondition ( SynchronizedCondition & sc );
         SynchronizedCondition& operator=( SynchronizedCondition & sc );

        /* \brief acquire the lock.
         */
         void lock()
         {
            _lock.acquire();
            memoryFence();
         }

      protected:

        /* \brief Sets a waiter Workdescriptor.
         */
         virtual void addWaiter( WorkDescriptor* wd) = 0;

        /* \brief Returns true if there's any waiter on the condition.
         */
         virtual bool hasWaiters() = 0;

        /* \brief Returns a waiter adn removes it from the condition.
         */
         virtual WorkDescriptor* getAndRemoveWaiter() = 0;
      
      public:
        /* \brief Constructor
         */
         SynchronizedCondition ( ConditionChecker *cc = NULL) : _lock(), _conditionChecker(cc) { }

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
         void setConditionChecker( ConditionChecker *cc )
         {
            if ( _conditionChecker != NULL )
               delete (_conditionChecker);
            _conditionChecker = cc;
         }

        /* \brief Release the lock. The wait() method can switch context
         * so it is necessary this function to be public so that the switchHelper
         * can unlock it after removing the current WD from the stack.
         */
         void unlock()
         {
            memoryFence();
            _lock.release();
         }
   };

  /* \brief SynchronizedCondition specialization that checks equality fon one
   * variable and with just one waiter.
   */
   class SingleSyncCond : public SynchronizedCondition
   {
      private:
         /**< Pointer to the WD waiting for the condition. */
         WorkDescriptor* _waiter;

         // Disable copy constructor and assign operator.
         SingleSyncCond ( SingleSyncCond & ssc );
         SingleSyncCond& operator=( SingleSyncCond & ssc );

      public:
        /* \brief Constructor 
         * \param var Variable which value is used for synchronization.
         * \param condition Value expected
         */
         SingleSyncCond ( ConditionChecker *cc = NULL) : SynchronizedCondition( cc ), _waiter( NULL ) { }

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
   class MultipleSyncCond : public SynchronizedCondition
   {
      private:
         /**< type vector of workdescriptors. */
         typedef std::vector<WorkDescriptor*> WorkDescriptorList;

         /**< List of WDs that wait on the condition. */
         WorkDescriptorList _waiters;

         // Disable copy constructor and assign operator
         MultipleSyncCond ( MultipleSyncCond & ssc );
         MultipleSyncCond& operator=( MultipleSyncCond & ssc );

      public:
        /* \brief Constructor 
         * \param var Variable which value is used for synchronization
         * \param condition Value expected
         */
         MultipleSyncCond (ConditionChecker *cc, size_t size) : SynchronizedCondition( cc ), _waiters()
         {
            _waiters.reserve(size);
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
