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

#include "asyncthread.hpp"

#include "schedule_decl.hpp"
#include "processingelement.hpp"
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
   ASYNC_THREAD_WAIT_INPUTS_EVENT,           /* 5 */
   ASYNC_THREAD_CP_DATA_IN_EVENT,            /* 6 */
   ASYNC_THREAD_CP_DATA_OUT_EVENT,           /* 7 */
   ASYNC_THREAD_CHECK_EVTS_EVENT,            /* 8 */
   ASYNC_THREAD_PROCESS_EVT_EVENT,           /* 9 */
   ASYNC_THREAD_SYNCHRONIZE_EVENT           /* 10 */
} AsyncThreadState_t;

bool AsyncThread::inlineWorkDependent( WD &work )
{
   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_INLINE_WORK_DEP_EVENT );

   debug( "[Async] At inlineWorkDependent, adding WD " << &work << " : " << work.getId() << " to running WDs list" );

   // Add WD to the queue
   _runningWDs.push_back( &work );
   _runningWDsCounter++;

#if PRINT_LIST
   std::stringstream s;
   s << "[Async]@inlineWorkDependent Running WDs: | ";
   for ( std::list<WD *>::iterator it = _runningWDs.begin(); it != _runningWDs.end(); it++ ) {
      WD * w = *it;
      s << w->getId() << " | ";
   }
   s << std::endl;
   std::cout << s.str();
#endif

   // Start steps to run this WD
   this->preRunWD( &work );

   ASYNC_THREAD_CLOSE_EVENT;

   return false;
}


void AsyncThread::idle()
{
   //ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_CHECK_EVTS_EVENT )
   checkEvents();
   //ASYNC_THREAD_CLOSE_EVENT

   WD * last = ( _runningWDsCounter != 0 ) ? _runningWDs.back() : getCurrentWD();

   while ( canGetWork() ) {
      // Fill WD's queue until we get the desired number of prefetched WDs
      WD * next = Scheduler::prefetch( ( BaseThread *) this, *last );

      if ( next != NULL ) {
         debug( "[Async] At idle, adding WD " << next << " : " << next->getId() << " to running WDs list" );

         // Add WD to the queue
         _runningWDs.push_back( next );
         _runningWDsCounter++;

#if PRINT_LIST
         std::stringstream s;
         s << "[Async]@idle Running WDs: | ";
         for ( std::list<WD *>::iterator it = _runningWDs.begin(); it != _runningWDs.end(); it++ ) {
            WD * w = *it;
            s << w->getId() << " | ";
         }
         s << std::endl;
         std::cout << s.str();
#endif

         // Start steps to run this WD
         this->preRunWD( next );

         //ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_CHECK_EVTS_EVENT )
         checkEvents();
         //ASYNC_THREAD_CLOSE_EVENT

         last = next;

      } else {
         // If no WD was returned, break the loop
         break;
      }
   }
}


void AsyncThread::preRunWD ( WD * wd )
{
   debug( "[Async] Prerunning WD " << wd << " : " << wd->getId() );

   _previousWD = this->getCurrentWD();

   this->setCurrentWD( *wd );

   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_PRE_RUN_EVENT );

   // This will start WD's copies
   wd->init();

   ASYNC_THREAD_CLOSE_EVENT;

   this->setCurrentWD( *_previousWD );
}


void AsyncThread::runWD ( WD * wd )
{
   // Check WD's dependencies
   if ( wd->getNumDepsPredecessors() != 0 ) {
      // Its WD predecessor is still running, so enqueue another event
      // to make this WD wait till the predecessor has finished
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

      return;
   }

   debug( "[Async] Running WD " << wd << " : " << wd->getId() );

   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_RUN_EVENT );

   // This will wait for WD's inputs
   wd->start( WD::IsNotAUserLevelThread );

   GenericEvent * evt = this->createRunEvent( wd );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " First event after WD is executed" );
#endif
   evt->setCreated();

   // Run WD
   //this->inlineWorkDependent( *wd );
   this->runWDDependent( *wd );

   evt->setPending();

#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " action:closeUsrFuncInstr" );
#endif

   Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD *>::MemFunPtr1 ) &AsyncThread::postRunWD, this, wd );
   evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " action:AsyncThread::postRunWD" );
#endif

   addEvent( evt );

   ASYNC_THREAD_CLOSE_EVENT;
}


void AsyncThread::postRunWD ( WD * wd )
{
   debug( "[Async] Postrunning WD " << wd << " : " << wd->getId() );

   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_POST_RUN_EVENT );

   wd->finish();

   _runningWDs.remove( wd );
   _runningWDsCounter--;

   debug( "[Async] Removing WD " << wd << " remaining WDs = " << _runningWDsCounter );

#if PRINT_LIST
   std::stringstream s;
   s << "[Async]@postRunWD Running WDs: | ";
   for ( std::list<WD *>::iterator it = _runningWDs.begin(); it != _runningWDs.end(); it++ ) {
      WD * w = *it;
      s << w->getId() << " | ";
   }
   s << std::endl;
   std::cout << s.str();
#endif

   ASYNC_THREAD_CLOSE_EVENT;
}



void AsyncThread::copyDataIn( WorkDescriptor& work )
{
   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_CP_DATA_IN_EVENT );

   // WD has no copies
   if ( work.getNumCopies() == 0 ) {
      GenericEvent * evt = this->createPreRunEvent( &work );
#ifdef NANOS_GENERICEVENT_DEBUG
      evt->setDescription( evt->getDescription() + " No inputs, event triggers runWD" );
#endif
     evt->setCreated();

     addEvent( evt );

     Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::runWD, this, &work );
     evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
     evt->setDescription( evt->getDescription() + " action:AsyncThread::runWD" );
#endif

     evt->setPending();
     evt->setRaised();

   } else {
      GenericEvent * lastEvt = NULL;
      CopyData *copies = work.getCopies();
      for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
         CopyData & cd = copies[i];
         cd.initCopyDescriptor();
         CopyDescriptor * cpdesc = cd.getCopyDescriptor();
         uint64_t tag = ( uint64_t ) cd.isPrivate() ? ( ( uint64_t ) work.getData() + ( unsigned long ) cd.getAddress() ) : cd.getAddress();

         GenericEvent * evt = this->createPreRunEvent( &work );
#ifdef NANOS_GENERICEVENT_DEBUG
         evt->setDescription( evt->getDescription() + " copy input " + toString<uint64_t>( tag ) );
#endif
         evt->setCreated();

         if ( cd.isInput() ) {
            NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("copy-in") );
            NANOS_INSTRUMENT( static nanos_event_value_t value = (nanos_event_value_t) cd.getSize() );
            NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &key, &value ) );
         }

         if ( cd.isPrivate() ) {
            runningOn()->registerPrivateAccessDependent( *( work.getParent()->getDirectory( true ) ), cd, tag  );
         } else {
            runningOn()->registerCacheAccessDependent( *( work.getParent()->getDirectory( true ) ), cd, tag );
         }

         evt->setPending();

         if ( cpdesc->_copying || cpdesc->_flushing ) {
            Action * action = new_action( ( ActionPtrMemFunPtr1<AsyncThread, CopyDescriptor>::PtrMemFunPtr1 ) &AsyncThread::synchronize, this, *( cd.getCopyDescriptor() ) );
            evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
            evt->setDescription( evt->getDescription() + " action:AsyncThread::synchronize" );
#endif
         }

         addEvent( evt );

         lastEvt = evt;
      }

      if ( lastEvt ) {
         Action * check = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::checkEvents, *this, &work );
         lastEvt->addNextAction( check );
#ifdef NANOS_GENERICEVENT_DEBUG
         lastEvt->setDescription( lastEvt->getDescription() + " action:AsyncThread::checkEventsWD" );
#endif
         Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::runWD, *this, &work );
         lastEvt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
        lastEvt->setDescription( lastEvt->getDescription() + " action:AsyncThread::runWD" );
#endif
      }
   }

   ASYNC_THREAD_CLOSE_EVENT;
}

void AsyncThread::waitInputs( WorkDescriptor &work )
{
   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_WAIT_INPUTS_EVENT );

   // Should never find a non-raised event
   for ( GenericEventList::iterator it = _pendingEvents.begin(); it != _pendingEvents.end(); it++ ) {
      GenericEvent * evt = *it;
      if ( evt->getWD() == &work && !( evt->isRaised() || evt->isCompleted() ) ) {
         // May not be an error in the case of GPUs since CUDA events are only checked every 100 times
         //warning( "Found a non-raised event! For WD " << evt->getWD()->getId() << " about " << evt->getDescription() );
         evt->waitForEvent();
      }
   }

   ASYNC_THREAD_CLOSE_EVENT;
}

void AsyncThread::copyDataOut( WorkDescriptor& work )
{
   ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_CP_DATA_OUT_EVENT );

   // WD has no copies
   if ( work.getNumCopies() == 0 ) {
      GenericEvent * evt = this->createPostRunEvent( &work );
#ifdef NANOS_GENERICEVENT_DEBUG
      evt->setDescription( evt->getDescription() + " No outputs, event triggers finishWork" );
#endif
      evt->setCreated();

      addEvent( evt );

      Action * action = new_action( ( ActionFunPtr2<WD *, bool>::FunPtr2 ) &Scheduler::finishWork, &work, false );
      evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
      evt->setDescription( evt->getDescription() + " action:Scheduler::finishWork" );
#endif

      evt->setPending();
      evt->setRaised();

   } else {
      GenericEvent * lastEvt = NULL;
      CopyData *copies = work.getCopies();
      for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
         CopyData & cd = copies[i];
         cd.initCopyDescriptor();
         CopyDescriptor * cpdesc = cd.getCopyDescriptor();
         uint64_t tag = ( uint64_t ) cd.isPrivate() ? ( ( uint64_t ) work.getData() + ( unsigned long ) cd.getAddress() ) : cd.getAddress();

         GenericEvent * evt = this->createPostRunEvent( &work );
#ifdef NANOS_GENERICEVENT_DEBUG
         evt->setDescription( evt->getDescription() + " copy output" );
#endif
         evt->setCreated();

         if ( cd.isOutput() ) {
            NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("copy-out") );
            NANOS_INSTRUMENT( static nanos_event_value_t value = (nanos_event_value_t) cd.getSize() );
            NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &key, &value ) );
         }

         if ( cd.isPrivate() ) {
            runningOn()->unregisterPrivateAccessDependent( *( work.getParent()->getDirectory( true ) ), cd, tag );
         } else {
            runningOn()->unregisterCacheAccessDependent( *( work.getParent()->getDirectory( true ) ), cd, tag, cd.isOutput() );

            // We need to create the directory in parent's parent if it does not exist. Otherwise, applications with
            // at least 3-level nesting tasks with the inner-most level being from a device with separate memory space
            // will fail because parent's parent directory is NULL
            if ( work.getParent()->getParent() != work.getParent() && work.getParent()->getParent() != NULL ) {
               Directory * dir = work.getParent()->getParent()->getDirectory( true );
               dir->updateCurrentDirectory( tag, *( work.getParent()->getDirectory( true ) ) );
            }
         }

         evt->setPending();

         if ( cpdesc->_copying || cpdesc->_flushing ) {
            Action * action = new_action( ( ActionPtrMemFunPtr1<AsyncThread, CopyDescriptor>::PtrMemFunPtr1 ) &AsyncThread::synchronize, this, *( cd.getCopyDescriptor() ) );
            evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
            evt->setDescription( evt->getDescription() + " action:AsyncThread::synchronize" );
#endif
         }

         addEvent( evt );

         lastEvt = evt;
      }

      if ( lastEvt ) {
         WorkDescriptor * actionWD = &work;
         Action * action = new_action( ( ActionFunPtr2<WD *, bool>::FunPtr2 ) &Scheduler::finishWork, actionWD, false );
         lastEvt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
         lastEvt->setDescription( lastEvt->getDescription() + " action:Scheduler::finishWork" );
#endif
      }
   }

   ASYNC_THREAD_CLOSE_EVENT;
}


void AsyncThread::synchronize( CopyDescriptor cd )
{
   //ASYNC_THREAD_CREATE_EVENT( ASYNC_THREAD_SYNCHRONIZE_EVENT );
   runningOn()->synchronize( cd );
   //ASYNC_THREAD_CLOSE_EVENT;
}

void AsyncThread::addEvent( GenericEvent * evt )
{
   _pendingEvents.push_back( evt );
   _pendingEventsCounter++;
}

WD * AsyncThread::getNextWD ()
{
   if ( canGetWork() ) return BaseThread::getNextWD();

   return NULL;
}

bool AsyncThread::hasNextWD ()
{
   if ( canGetWork() ) return BaseThread::hasNextWD();

   return false;
}
