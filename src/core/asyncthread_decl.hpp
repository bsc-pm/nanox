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

#ifndef _ASYNC_THREAD_DECL
#define _ASYNC_THREAD_DECL

#include <list>

#include "basethread_decl.hpp"
#include "genericevent_decl.hpp"


#include "workdescriptor_fwd.hpp"
#include "processingelement_fwd.hpp"
#include "debug.hpp"
#include "atomic_decl.hpp"
#include "schedule_fwd.hpp"
#include "threadteam_fwd.hpp"
#include "allocator_decl.hpp"

namespace nanos {
   class AsyncThread : public BaseThread
   {
      public:
         typedef std::list<WD *> WDList;
         typedef std::deque<GenericEvent *> GenericEventList;

      private:
         std::list<WD *>   _runningWDs;
         unsigned int      _runningWDsCounter;

         GenericEventList  _pendingEvents;
         unsigned int      _pendingEventsCounter;

         unsigned int      _recursiveCounter;

         // Previous running WD, used for instrumentation only
         WD *              _previousWD; 

      private:
        /*! \brief AsyncThread default constructor
         */
         AsyncThread ();

        /*! \brief AsyncThread copy constructor (private)
         */
         AsyncThread( const AsyncThread & );

        /*! \brief AsyncThread copy assignment operator (private)
         */
         const AsyncThread & operator= ( const AsyncThread & );

      public:
        /*! \brief AsyncThread constructor
         */
         AsyncThread ( unsigned int osId, WD &wd, ProcessingElement *creator = 0 );

        /*! \brief AsyncThread destructor
         */
         virtual ~AsyncThread()
         {
            ensure0( _runningWDs.empty(), "WD list not empty in AsyncThread!" );
            ensure0( _runningWDsCounter == 0, "Running WD list counter not 0 in AsyncThread!" );
            ensure0( _pendingEvents.empty(), "Event list not empty in AsyncThread!" );
            ensure0( _pendingEventsCounter == 0, "Event list counter not 0 in AsyncThread!" );
         }


         // TODO: Consider if we need them
         virtual void initializeDependent( void ) {}
         //virtual void runDependent ( void );

         virtual bool inlineWorkDependent ( WD &work );
         virtual bool runWDDependent ( WD &work, GenericEvent * evt ) = 0;

         // Must be implemented by children classes
         //virtual bool inlineWorkDependent( WD &work );

         virtual void yield() { this->idle(); }

         virtual void idle( bool dummy = false );

         virtual void processTransfers ();

         virtual void preRunWD ( WD * wd );
         virtual void runWD ( WD * wd );
         virtual void postRunWD ( WD * wd );

         virtual void checkWDInputs( WD * wd );
         virtual void checkWDOutputs( WD * wd );

         virtual bool processNonAllocatedWDData ( WD * wd );
         virtual bool processDependentWD ( WD * wd );

         virtual GenericEvent * createPreRunEvent( WD * wd ) = 0;
         virtual GenericEvent * createRunEvent( WD * wd ) = 0;
         virtual GenericEvent * createPostRunEvent( WD * wd ) = 0;

         virtual void closeWDEvent() {}

         // TODO: Do we need a getDeviceId()?
         //virtual int getGPUDevice ();


         //virtual void switchTo( WD *work, SchedulerHelper *helper );
         //virtual void exitTo( WD *work, SchedulerHelper *helper );

         //virtual void switchHelperDependent( WD* oldWD, WD* newWD, void *arg );
         //virtual void exitHelperDependent( WD* oldWD, WD* newWD, void *arg ) {};

         // Should bind be implemented?
         //virtual void bind( void );

         virtual void addEvent( GenericEvent * evt );
         const AsyncThread::GenericEventList& getEvents( );
         virtual void checkEvents();
         void checkEvents( WD * wd );

         virtual bool canGetWork ();


         // Additional checkings for next WDs
         void addNextWD ( WD *next );
         WD * getNextWD ();
         bool hasNextWD () const;

         // Set whether the thread will schedule WDs or not used by getImmediateSuccessor()
         // If so, WD's dependencies should be kept till WD is finished
         virtual bool keepWDDeps() { return true; }

   };
} // namespace nanos

#endif //_ASYNC_THREAD_DECL
