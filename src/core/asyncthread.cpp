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

using namespace nanos;


#define PRINT_LIST 0


bool AsyncThread::inlineWorkDependent( WD &work )
{
   //AsyncThread::runWD( &work );

   //_runningWDs.push_back( &work );
   //_runningWDsCounter++;

   debug( "[Async] At inlineWorkDependent, adding WD " << &work << " : " << work.getId() << " to running WDs list" );

   // Add WD to the queue
   //addNextWD( &work );

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

   return false;
}


void AsyncThread::idle()
{
   checkEvents();

   WD * last = ( _runningWDsCounter != 0 ) ? _runningWDs.back() : getCurrentWD();

   while ( canGetWork() ) {
      // Fill WD's queue until we get the desired number of prefetched WDs
      WD * next = Scheduler::prefetch( ( BaseThread *) this, *last );

      if ( next != NULL ) {

         debug( "[Async] At idle, adding WD " << next << " : " << next->getId() << " to running WDs list" );

         // Add WD to the queue
         //addNextWD( next );
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

         checkEvents();

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

   WD *oldwd = this->getCurrentWD();
   this->setCurrentWD( *wd );
   // Instrumenting context switch: oldwd leaves CPU, but will come back later (last = false)
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( oldwd, wd, /* last */ false ) );

   // This will start WD's copies
   wd->init();

   this->setCurrentWD( *oldwd );
   // Instrumenting context switch: wd leaves CPU, but will come back later (last = false)
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( wd, oldwd, /* last */ false ) );
}


void AsyncThread::runWD ( WD * wd )
{
   // Check WD's dependencies
   if ( wd->getNumDepsPredecessors() != 0 ) {
      // Its WD predecessor is still running, so enqueue another event
      // to make this WD wait till the predecessor has finished
      GenericEvent * deps;

#ifdef NANOS_DEBUG_ENABLED
      deps = NEW GenericEvent( wd, "Checking WD deps" );
#else
      deps = NEW GenericEvent( wd );
#endif

      Action * depsAction = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::runWD, *this, wd );
      deps->addNextAction( depsAction );
#ifdef NANOS_DEBUG_ENABLED
      deps->setDescription( deps->getDescription() + " action:AsyncThread::runWD" );
#endif

      deps->setRaised();

      // WARNING: push it front to avoid deadlocks
      _pendingEvents.push_front( deps );
      _pendingEventsCounter++;

      return;
   }

   debug( "[Async] Running WD " << wd << " : " << wd->getId() );

   WD *oldwd = this->getCurrentWD();
   this->setCurrentWD( *wd );
   // Instrumenting context switch: oldwd leaves CPU, but will come back later (last = false)
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( oldwd, wd, /* last */ false ) );

   // This will wait for WD's inputs
   wd->start( WD::IsNotAUserLevelThread );

   GenericEvent * evt = this->createRunEvent( wd );
#ifdef NANOS_DEBUG_ENABLED
   evt->setDescription( evt->getDescription() + " First event after WD is executed" );
#endif
   evt->setCreated();

   // Run WD
   //this->inlineWorkDependent( *wd );
   this->runWDDependent( *wd );

   evt->setPending();

   Action * instrument = new_action( ( ActionMemFunPtr1<AsyncThread, WD *>::MemFunPtr1 ) &AsyncThread::raiseWDClosingEvents, this, wd );
   evt->addNextAction( instrument );
#ifdef NANOS_DEBUG_ENABLED
   evt->setDescription( evt->getDescription() + " action:closeInstr" );
#endif

   Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD *>::MemFunPtr1 ) &AsyncThread::postRunWD, this, wd );
   evt->addNextAction( action );
#ifdef NANOS_DEBUG_ENABLED
   evt->setDescription( evt->getDescription() + " action:AsyncThread::postRunWD" );
#endif

   addEvent( evt );

   this->setCurrentWD( *oldwd );
   // Instrumenting context switch: wd leaves CPU, but will come back later (last = false)
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( wd, oldwd, /* last */ false ) );
}


void AsyncThread::postRunWD ( WD * wd )
{
   debug( "[Async] Postrunning WD " << wd << " : " << wd->getId() );

   WD *oldwd = this->getCurrentWD();
   this->setCurrentWD( *wd );
   // Instrumenting context switch: oldwd leaves CPU, but will come back later (last = false)
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( oldwd, wd, /* last */ false ) );

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

   this->setCurrentWD( *oldwd );
   // Instrumenting context switch: wd leaves cpu and will not come back (last = true) and oldwd enters
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( wd, oldwd, /* last */ true ) );

}



void AsyncThread::copyDataIn( WorkDescriptor& work )
{
   // WD has no copies
   if ( work.getNumCopies() == 0 ) {
      GenericEvent * evt = this->createPreRunEvent( &work );
#ifdef NANOS_DEBUG_ENABLED
      evt->setDescription( evt->getDescription() + " No inputs, event triggers runWD" );
#endif
     evt->setCreated();

     addEvent( evt );

      Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::runWD, this, &work );
      evt->addNextAction( action );
#ifdef NANOS_DEBUG_ENABLED
      evt->setDescription( evt->getDescription() + " action:AsyncThread::runWD" );
#endif

      evt->setPending();
      evt->setRaised();

   } else {
      GenericEvent * lastEvt = NULL;
      CopyData *copies = work.getCopies();
      for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
         CopyData & cd = copies[i];
         uint64_t tag = ( uint64_t ) cd.isPrivate() ? ( ( uint64_t ) work.getData() + ( unsigned long ) cd.getAddress() ) : cd.getAddress();

         GenericEvent * evt = this->createPreRunEvent( &work );
   #ifdef NANOS_DEBUG_ENABLED
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

         CopyDescriptor cpdesc = cd.getCopyDescriptor();

         if ( cpdesc.copying || cpdesc.flushing ) {
            Action * action = new_action( ( ActionPtrMemFunPtr1<AsyncThread, CopyDescriptor>::PtrMemFunPtr1 ) &AsyncThread::synchronize, this, cd.getCopyDescriptor() );
            evt->addNextAction( action );
#ifdef NANOS_DEBUG_ENABLED
            evt->setDescription( evt->getDescription() + " action:AsyncThread::synchronize" );
#endif
         }

         addEvent( evt );

         lastEvt = evt;
      }

      if ( lastEvt ) {
         Action * check = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::checkEvents, *this, &work );
         lastEvt->addNextAction( check );
   #ifdef NANOS_DEBUG_ENABLED
         lastEvt->setDescription( lastEvt->getDescription() + " action:AsyncThread::checkEventsWD" );
   #endif
         Action * action = new_action( ( ActionMemFunPtr1<AsyncThread, WD*>::MemFunPtr1 ) &AsyncThread::runWD, *this, &work );
         lastEvt->addNextAction( action );
   #ifdef NANOS_DEBUG_ENABLED
        lastEvt->setDescription( lastEvt->getDescription() + " action:AsyncThread::runWD" );
   #endif
      }
   }
}

void AsyncThread::waitInputs( WorkDescriptor &work )
{
   // Should never find a non-raised event
   for ( GenericEventList::iterator it = _pendingEvents.begin(); it != _pendingEvents.end(); it++ ) {
      GenericEvent * evt = *it;
      if ( evt->getWD() == &work && !( evt->isRaised() || evt->isCompleted() ) ) {
         // May not be an error in the case of GPUs since CUDA events are only checked every 100 times
         //warning( "Found a non-raised event! For WD " << evt->getWD()->getId() << " about " << evt->getDescription() );
         evt->waitForEvent();
      }
   }
}

void AsyncThread::copyDataOut( WorkDescriptor& work )
{
   // WD has no copies
   if ( work.getNumCopies() == 0 ) {
      GenericEvent * evt = this->createPostRunEvent( &work );
#ifdef NANOS_DEBUG_ENABLED
      evt->setDescription( evt->getDescription() + " No outputs, event triggers finishWork" );
#endif
      evt->setCreated();

      addEvent( evt );

      Action * action = new_action( ( ActionFunPtr2<WD *, WD *>::FunPtr2 ) &Scheduler::finishWork, ( WD * ) NULL, &work );
      evt->addNextAction( action );
#ifdef NANOS_DEBUG_ENABLED
      evt->setDescription( evt->getDescription() + " action:Scheduler::finishWork" );
#endif

      evt->setPending();
      evt->setRaised();
   } else {
      GenericEvent * lastEvt = NULL;
      CopyData *copies = work.getCopies();
      for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
         CopyData & cd = copies[i];
         uint64_t tag = ( uint64_t ) cd.isPrivate() ? ( ( uint64_t ) work.getData() + ( unsigned long ) cd.getAddress() ) : cd.getAddress();

         GenericEvent * evt = this->createPostRunEvent( &work );
   #ifdef NANOS_DEBUG_ENABLED
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

         CopyDescriptor cpdesc = cd.getCopyDescriptor();

         if ( cpdesc.copying || cpdesc.flushing ) {
            Action * action = new_action( ( ActionPtrMemFunPtr1<AsyncThread, CopyDescriptor>::PtrMemFunPtr1 ) &AsyncThread::synchronize, this, cd.getCopyDescriptor() );
            evt->addNextAction( action );
#ifdef NANOS_DEBUG_ENABLED
            evt->setDescription( evt->getDescription() + " action:AsyncThread::synchronize" );
#endif
         }

         addEvent( evt );

         lastEvt = evt;
      }

      if ( lastEvt ) {
         Action * action = new_action( ( ActionFunPtr2<WD *, WD *>::FunPtr2 ) &Scheduler::finishWork, ( WD * ) NULL, &work );
         lastEvt->addNextAction( action );
   #ifdef NANOS_DEBUG_ENABLED
         lastEvt->setDescription( lastEvt->getDescription() + " action:Scheduler::finishWork" );
   #endif
      }
   }
}


void AsyncThread::synchronize( CopyDescriptor cd )
{
   runningOn()->synchronize( cd );
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
