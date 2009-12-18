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
#include "atomic.hpp"
#include "debug.hpp"

namespace nanos
{
   class WorkDescriptor;

  /*! \brief Abstract synchronization class.
   */
   class SynchronizedCondition
   {
      private:
         /**< Lock to block and unblock WorkDescriptors securely*/
         Lock _lock;

         // Disable copy constructor and assign operator
         SynchronizedCondition ( SynchronizedCondition & sc );
         SynchronizedCondition& operator=( SynchronizedCondition & sc );

        /* \brief acquire the lock
         */
         void lock()
         {
            _lock.acquire();
            memoryFence();
         }

      protected:

        /* \brief Must return true if the condition for the synchronization is satisfied
         */
         virtual bool checkCondition() = 0;

        /* \brief Sets a waiter Workdescriptor
         */
         virtual void setWaiter( WorkDescriptor* wd) = 0;

        /* \brief Returns true if there's any waiter on the condition
         */
         virtual bool hasWaiters() = 0;

        /* \brief Returns a waiter adn removes it from the condition
         */
         virtual WorkDescriptor* getAndRemoveWaiter() = 0;
      
      public:
        /* \brief Constructor
         */
         SynchronizedCondition () : _lock() { }

        /* \brief virtual destructor
         */
         virtual ~SynchronizedCondition() { }

         void wait();
         void signal();

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
     variable and with just one waiter.
   */
   template<typename T>
   class SingleSyncCond : public SynchronizedCondition
   {
      private:
         WorkDescriptor* _waiter;
         volatile T*     _var;
         T               _condition;

         // Disable copy constructor and assign operator
         SingleSyncCond ( SingleSyncCond & ssc );
         SingleSyncCond& operator=( SingleSyncCond & ssc );

      public:
        /* \brief Constructor 
         * \param var Variable which value is used for synchronization
         * \param condition Value expected
         */
         SingleSyncCond ( T* var, const T &condition ) : SynchronizedCondition(), 
            _waiter( NULL ), _var( var ), _condition( condition ) { }

         SingleSyncCond ( volatile T* var, const T &condition ) : SynchronizedCondition(), 
            _waiter( NULL ), _var( var ), _condition( condition ) { }

        /* \brief virtual destructor
         */
         virtual ~SingleSyncCond() { }

      protected:
        /* \brief Checks equality between the variable and the condition with
         * which it has been created.
         */
         virtual bool checkCondition()
         {
            return (*_var == _condition);
         }

        /* \brief Sets the waiter to wd
         */
         virtual void setWaiter( WorkDescriptor* wd )
         {
            _waiter = wd;
         }

        /* \brief Returns true if there's a waiter
         */
         virtual bool hasWaiters()
         {
            return _waiter != NULL;
         }

        /* \brief Returns the waiter and sets it to NULL
         */
         virtual WorkDescriptor* getAndRemoveWaiter()
         {
            WorkDescriptor* result = _waiter;
            _waiter = NULL;
            return result;
         }
   };

  /* \brief SynchronizedCondition specialization that checks equality fon one
     variable and with more than one waiter.
   */
   template<typename T>
   class MultipleSyncCond : public SynchronizedCondition
   {
      private:
         typedef std::list<WorkDescriptor *> WorkDescriptorList;

         WorkDescriptorList _waiters;
         volatile T*     _var;
         T               _condition;

         // Disable copy constructor and assign operator
         MultipleSyncCond ( MultipleSyncCond & ssc );
         MultipleSyncCond& operator=( MultipleSyncCond & ssc );

      public:
        /* \brief Constructor 
         * \param var Variable which value is used for synchronization
         * \param condition Value expected
         */
         MultipleSyncCond ( T* var, const T &condition ) : SynchronizedCondition(), 
            _waiters(), _var( var ), _condition( condition ) { }

         MultipleSyncCond ( volatile T* var, const T &condition ) : SynchronizedCondition(), 
            _waiters(), _var( var ), _condition( condition ) { }

        /* \brief virtual destructor
         */
         virtual ~MultipleSyncCond() { }

      protected:
        /* \brief Checks equality between the variable and the condition with
         * which it has been created.
         */
         virtual bool checkCondition()
         {
            return (*_var == _condition);
         }

        /* \brief Sets the waiter to wd
         */
         virtual void setWaiter( WorkDescriptor* wd )
         {
            _waiters.push_back( wd );
         }

        /* \brief Returns true if there's a waiter
         */
         virtual bool hasWaiters()
         {
            return !( _waiters.empty() );
         }

        /* \brief Returns the waiter and sets it to NULL
         */
         virtual WorkDescriptor* getAndRemoveWaiter()
         {
            if ( _waiters.empty() )
               return NULL;
            WorkDescriptor* reslt = *( _waiters.begin() );
            _waiters.pop_front();
            return reslt;
         }
   };
}

#endif
