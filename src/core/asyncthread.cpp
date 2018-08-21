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

#include "asyncthread.hpp"

#include "schedule_decl.hpp"
#include "processingelement.hpp"
#include "workdescriptor_decl.hpp"
//#include "system.hpp"
//#include "synchronizedcondition.hpp"
#include <unistd.h>


using namespace nanos;


#define PRINT_LIST 0

// Macro's to instrument the code and make it cleaner
#define ASYNC_THREAD_CREATE_EVENT(x)   NANOS_INSTRUMENT( \
		sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "async-thread" ), (x) ); )

#define ASYNC_THREAD_CLOSE_EVENT       NANOS_INSTRUMENT( \
		sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "async-thread" ), 0 ); )


typedef enum {
   ASYNC_THREAD_NULL_EVENT,                  /* 0 */
   ASYNC_THREAD_INLINE_WORK_DEP_EVENT,       /* 1 */
   ASYNC_THREAD_PRE_RUN_EVENT,               /* 2 */
   ASYNC_THREAD_RUN_EVENT,                   /* 3 */
   ASYNC_THREAD_POST_RUN_EVENT,              /* 4 */
   ASYNC_THREAD_SCHEDULE_EVENT,              /* 5 */
//   ASYNC_THREAD_WAIT_INPUTS_EVENT,           /* 5 */
   ASYNC_THREAD_CHECK_WD_INPUTS_EVENT,       /* 6 */
   ASYNC_THREAD_CHECK_WD_OUTPUTS_EVENT,      /* 7 */
   ASYNC_THREAD_CP_DATA_IN_EVENT,            /* 8 */
   ASYNC_THREAD_CP_DATA_OUT_EVENT,           /* 9 */
   ASYNC_THREAD_CHECK_EVTS_EVENT,           /* 10 */
   ASYNC_THREAD_PROCESS_EVT_EVENT,          /* 11 */   /* WARNING!! Value hard-coded in asyncthread.hpp */
   ASYNC_THREAD_SYNCHRONIZE_EVENT,          /* 12 */
} AsyncThreadState_t;


AsyncThread::AsyncThread ( unsigned int osId, WD &wd, ProcessingElement *creator ) :
      BaseThread( osId, wd, creator ), _runningWDs(), _runningWDsCounter( 0 ),
      _pendingEvents(), _pendingEventsCounter( 0 ), _recursiveCounter( 0 ),
      _previousWD( NULL )
{
   sys.setPredecessorLists( true );
}


bool AsyncThread::inlineWorkDependent( WD &work )
{
   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_INLINE_WORK_DEP_EVENT );

   debug( "[Async] At inlineWorkDependent, adding WD " << &work << " : " << work.getId() << " to running WDs list" );

   // Add WD to the queue
   addNextWD( &work );

   ASYNC_THREAD_CLOSE_EVENT;

   return false;
}


void AsyncThread::idle( bool dummy )
{
   NANOS_INSTRUMENT( if ( _pendingEventsCounter > 0 ) { )
   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_CHECK_EVTS_EVENT );
   NANOS_INSTRUMENT( } )
   checkEvents();
   NANOS_INSTRUMENT( if ( _pendingEventsCounter > 0 ) { )
   ASYNC_THREAD_CLOSE_EVENT;
   NANOS_INSTRUMENT( } )

   WD * last = ( _runningWDsCounter != 0 ) ? _runningWDs.back() : getCurrentWD();
   WD * next = NULL;

   //ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_SCHEDULE_EVENT );

   while ( canGetWork() ) {

      NANOS_INSTRUMENT( if ( _pendingEventsCounter > 0 ) { )
      ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_CHECK_EVTS_EVENT );
      NANOS_INSTRUMENT( } )
      checkEvents();
      NANOS_INSTRUMENT( if ( _pendingEventsCounter > 0 ) { )
      ASYNC_THREAD_CLOSE_EVENT;
      NANOS_INSTRUMENT( } )

      if ( next == NULL ) {
         last = ( _runningWDsCounter != 0 ) ? _runningWDs.back() : getCurrentWD();
      }

      // Fill WD's queue until we get the desired number of prefetched WDs
      next = Scheduler::prefetch( ( BaseThread *) this, *last );

      if ( next != NULL ) {
         debug( "[Async] At idle, adding WD " << next << " : " << next->getId() << " to running WDs list" );

         // Add WD to the queue
         addNextWD( next );

         NANOS_INSTRUMENT( if ( _pendingEventsCounter > 0 ) { )
         ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_CHECK_EVTS_EVENT );
         NANOS_INSTRUMENT( } )
         checkEvents();
         NANOS_INSTRUMENT( if ( _pendingEventsCounter > 0 ) { )
         ASYNC_THREAD_CLOSE_EVENT;
         NANOS_INSTRUMENT( } )

         last = next;

      } else {
         // If no WD was returned, break the loop
         break;
      }
   }

   //ASYNC_THREAD_CLOSE_EVENT;
}

bool AsyncThread::processNonAllocatedWDData ( WD * wd )
{
   GenericEvent * event;

#ifdef NANOS_GENERICEVENT_DEBUG
   event = NEW GenericEvent( wd, "Data allocation failed" );
#else
   event = NEW GenericEvent( wd );
#endif

   Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::preRunWD, *this, wd );
   event->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
   event->setDescription( event->getDescription() + " action:AsyncThread::preRunWD" );
#endif

   event->setRaised();

   addEvent( event );

   return true;
}

void AsyncThread::preRunWD ( WD * wd )
{
   debug( "[Async] Prerunning WD " << wd << " : " << wd->getId() );

   _previousWD = getCurrentWD();
   setCurrentWD( *wd );

   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_PRE_RUN_EVENT );

   GenericEvent * evt = this->createPreRunEvent( wd );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " AsyncThread::preRunWD" );
#endif
   evt->setCreated();

   // It should be already done, but just in case...
   wd->_mcontrol.preInit();

   if ( !wd->started() ) {
      if ( !wd->_mcontrol.isMemoryAllocated() ) {
         wd->_mcontrol.initialize( *( this->runningOn() ) );
         if ( wd->_mcontrol.allocateTaskMemory() == false ) {
            // Data could not be allocated
            if ( processNonAllocatedWDData( wd ) ) {
               // Will try it later
               ASYNC_THREAD_CLOSE_EVENT;
               setCurrentWD( *_previousWD );
               return;
            }
         }
      }

      // This will start WD's copies
      wd->init();
      wd->preStart( WD::IsNotAUserLevelThread );
   }

   evt->setPending();

   Action * check = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::checkWDInputs, *this, wd );
   evt->addNextAction( check );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " action:AsyncThread::checkWDInputs" );
#endif

   addEvent( evt );

   ASYNC_THREAD_CLOSE_EVENT;

   setCurrentWD( *_previousWD );
}


void AsyncThread::checkWDInputs( WD * wd )
{
   debug( "[Async] Checking inputs of WD " << wd << " : " << wd->getId() );
   // Check if WD's inputs have already been copied

   _previousWD = getCurrentWD();
   setCurrentWD( *wd );

   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_CHECK_WD_INPUTS_EVENT );

#if 0
   // Should never find a non-raised event
   for ( GenericEventList::iterator it = _pendingEvents.begin(); it != _pendingEvents.end(); it++ ) {
      GenericEvent * evt = *it;
      if ( evt->getWD() == wd && !( evt->isRaised() || evt->isCompleted() ) ) {
         // May not be an error in the case of GPUs since CUDA events are only checked every 100 times
         //warning( "Found a non-raised event! For WD " << evt->getWD()->getId() << " about " << evt->getDescription() );
         evt->waitForEvent();
      }
   }
#endif

   if ( wd->isInputDataReady() ) {
      // Input data has been copied, we can run the WD
      runWD( wd );
   } else {
#if 0
      // Input data is not ready yet, we must wait
      GenericEvent * evt;

#ifdef NANOS_GENERICEVENT_DEBUG
      evt = NEW GenericEvent( wd, "Checking WD inputs" );
#else
      evt = NEW GenericEvent( wd );
#endif

      Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::checkWDInputs, *this, wd );
      evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
      evt->setDescription( evt->getDescription() + " action:AsyncThread::checkWDInputs" );
#endif

      evt->setRaised();

      addEvent( evt );
#else
      // Input data is not ready yet, we must wait
      CustomEvent * evt;
      Condition * cond = new_condition( ( ConditionPtrMemFunPtr0<WD>::PtrMemFunPtr0 ) &WD::isInputDataReady, wd );

#ifdef NANOS_GENERICEVENT_DEBUG
      evt = NEW CustomEvent( wd, "CustomEvt Checking WD inputs: WD::isInputDataReady", cond );
#else
      evt = NEW CustomEvent( wd, cond );
#endif

      Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::runWD, *this, wd );
      evt->addNextAction( action );

      evt->setPending();

      addEvent( evt );

#endif
   }

   ASYNC_THREAD_CLOSE_EVENT;

   setCurrentWD( *_previousWD );
}

bool AsyncThread::processDependentWD ( WD * wd )
{
   GenericEvent * deps;

#ifdef NANOS_GENERICEVENT_DEBUG
   deps = NEW GenericEvent( wd, "Checking WD deps" );
#else
   deps = NEW GenericEvent( wd );
#endif

   Action * depsAction = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::runWD, *this, wd );
   deps->addNextAction( depsAction );
#ifdef NANOS_GENERICEVENT_DEBUG
   deps->setDescription( deps->getDescription() + " action:AsyncThread::runWD" );
#endif

   deps->setRaised();

   addEvent( deps );

   return true;
}


void AsyncThread::runWD ( WD * wd )
{
   // Check WD's dependencies
   if ( wd->hasDepsPredecessors() ) {
      // Its WD predecessor is still running, so enqueue another event
      // to make this WD wait till the predecessor has finished

      bool waitDeps = processDependentWD( wd );

      if ( waitDeps ) return;
   }

   debug( "[Async] Running WD " << wd << " : " << wd->getId() );

   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_RUN_EVENT );

   GenericEvent * evt = this->createRunEvent( wd );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " First event after WD is executed" );
#endif
   evt->setCreated();

   // Run WD
   //this->inlineWorkDependent( *wd );
   this->runWDDependent( *wd, evt );

   evt->setPending();

#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " action:closeUsrFuncInstr" );
#endif

   Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD *>::MemFunPtr1 ) &AsyncThread::checkWDOutputs, this, wd );
   evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " action:AsyncThread::checkWDOutputs" );
#endif

   addEvent( evt );
   debug( "[Async] Finished running WD " << wd << " : " << wd->getId() );

   ASYNC_THREAD_CLOSE_EVENT;
}


void AsyncThread::checkWDOutputs( WD * wd )
{
   // Marks task event as finished
   NANOS_INSTRUMENT( closeWDEvent(); );

   // Check if WD's outputs have already been copied (if needed)
   _previousWD = getCurrentWD();
   setCurrentWD( *wd );

   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_CHECK_WD_OUTPUTS_EVENT );

   GenericEvent * evt = this->createPostRunEvent( wd );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " AsyncThread::checkWDOutputs" );
#endif
   evt->setCreated();

   wd->preFinish();

   evt->setPending();

   Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD *>::MemFunPtr1 ) &AsyncThread::postRunWD, this, wd );
   evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " action:AsyncThread::postRunWD" );
#endif

   addEvent( evt );

   ASYNC_THREAD_CLOSE_EVENT;

   setCurrentWD( *_previousWD );
}


void AsyncThread::postRunWD ( WD * wd )
{
   debug( "[Async] Postrunning WD " << wd << " : " << wd->getId() );

   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_POST_RUN_EVENT );

   if ( !wd->isOutputDataReady() ) {
#if 0
      // Output data is not ready yet, we must wait
      GenericEvent * evt;

#ifdef NANOS_GENERICEVENT_DEBUG
      evt = NEW GenericEvent( wd, "Checking WD outputs" );
#else
      evt = NEW GenericEvent( wd );
#endif

      Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::postRunWD, *this, wd );
      evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
      evt->setDescription( evt->getDescription() + " action:AsyncThread::checkWDOutputs" );
#endif

      evt->setRaised();

      addEvent( evt );

      ASYNC_THREAD_CLOSE_EVENT;

      return;
#else
      CustomEvent * evt;
      Condition * cond = new_condition( ( ConditionPtrMemFunPtr0<WD>::PtrMemFunPtr0 ) &WD::isOutputDataReady, wd );

#ifdef NANOS_GENERICEVENT_DEBUG
      evt = NEW CustomEvent( wd, "CustomEvt Checking WD outputs: WD::isOutputDataReady", cond );
#else
      evt = NEW CustomEvent( wd, cond );
#endif

      evt->setPending();

      Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD *>::MemFunPtr1 ) &AsyncThread::postRunWD, this, wd );
      evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
      evt->setDescription( evt->getDescription() + " action:AsyncThread::postRunWD" );
#endif

      addEvent( evt );

      ASYNC_THREAD_CLOSE_EVENT;

      return;
#endif
   }

   // Reaching this point means that output data has been copied, we can clean-up the WD
   _runningWDs.remove( wd );
   _runningWDsCounter--;
   //_runningWDsCounter = _runningWDs.size();

   ensure( _runningWDsCounter == _runningWDs.size(), "Running WDs counter doesn't match runningWDs list size!!" );

   debug( "[Async] Removing WD " << wd << " remaining WDs = " << _runningWDsCounter );

#if PRINT_LIST
   std::stringstream s;
   s << "[" << getId() << "][Async]@postRunWD Running WDs: | ";
   for ( std::list<WD *>::iterator it = _runningWDs.begin(); it != _runningWDs.end(); it++ ) {
      WD * w = *it;
      s << w->getId() << " | ";
   }
   s << std::endl;
   std::cout << s.str();
#endif

   // This can freeze as we can come from an invalidation, and then issue an allocation
   //Scheduler::finishWork( wd, canGetWork() );
   Scheduler::finishWork( wd, false );

   ASYNC_THREAD_CLOSE_EVENT;
}


void AsyncThread::addEvent( GenericEvent * evt )
{
   _pendingEvents.push_back( evt );
   _pendingEventsCounter++;
}

const AsyncThread::GenericEventList& AsyncThread::getEvents( ) {
    return _pendingEvents;
}

void AsyncThread::addNextWD ( WD *next )
{
   if ( next != NULL ) {
      debug( "[Async] Adding next WD " << next << " : " << next->getId() << " to running WDs list" );

      // Add WD to the queue
      _runningWDs.push_back( next );
      _runningWDsCounter++;

      ensure( _runningWDsCounter == _runningWDs.size(), "Running WDs counter doesn't match runningWDs list size!!" );

#if PRINT_LIST
      std::stringstream s;
      s << "[" << getId() << "][Async]@addNextWD (just added " << next->getId() << ") Running WDs: | ";
      for ( std::list<WD *>::iterator it = _runningWDs.begin(); it != _runningWDs.end(); it++ ) {
         WD * w = *it;
         s << w->getId() << " | ";
      }
      s << std::endl;
      std::cout << s.str();
#endif

      // Start steps to run this WD
      preRunWD( next );
   }
}

WD * AsyncThread::getNextWD ()
{
   //if ( canGetWork() ) return BaseThread::getNextWD();
   return NULL;
}

bool AsyncThread::hasNextWD () const
{
   //if ( canGetWork() ) return BaseThread::hasNextWD();
   return false;
}
