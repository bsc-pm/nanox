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

#include "instrumentation.hpp"

#include "instrumentationcontext.hpp"
#include "system.hpp"
#include "compatibility.hpp"
#include "workdescriptor.hpp"
#include "basethread.hpp"
#include <alloca.h>

using namespace nanos;

#ifdef NANOS_INSTRUMENTATION_ENABLED

/* ************************************************************************** */
/* ***                   C R E A T I N G   E V E N T S                    *** */
/* ************************************************************************** */
void Instrumentation::createStateEvent( Event *e, nanos_event_state_value_t state, InstrumentationContextData *icd )
{
   /* Registering a state event in instrucmentor context */
   if ( icd == NULL ) icd = myThread->getCurrentWD()->getInstrumentationContextData();
   _instrumentationContext.pushState(icd, state);

   /* Creating a state event */
   new (e) State(NANOS_STATE_START, state);
}

void Instrumentation::returnPreviousStateEvent ( Event *e, InstrumentationContextData *icd  )
{
   /* Recovering a state event in instrumentation context */
   if ( icd == NULL ) icd = myThread->getCurrentWD()->getInstrumentationContextData();

   /* Getting top of state stack: before pop (for stacked states backends) */
   nanos_event_state_value_t state = _instrumentationContext.topState( icd ); 

   /* Stack Pop */
   _instrumentationContext.popState( icd );

   /* Creating a state event */
   if ( _instrumentationContext.showStackedStates () || !_instrumentationContext.isContextSwitchEnabled() ) {
      new (e) State(NANOS_STATE_END,state);
   } else {
      /* Getting top of state stack: after pop (for non-stacked states backends) */
      state = _instrumentationContext.topState( icd ); 
      new (e) State(NANOS_STATE_START,state);
   }
}

void Instrumentation::createBurstEvent ( Event *e, nanos_event_key_t key, nanos_event_value_t value, InstrumentationContextData *icd )
{
   /* Recovering a state event in instrumentation context */
   if ( icd == NULL ) icd = myThread->getCurrentWD()->getInstrumentationContextData();

   /* Creating burst  event */
   new (e) Burst( true, key, value );

   _instrumentationContext.insertBurst( icd, *e );
}

void Instrumentation::closeBurstEvent ( Event *e, nanos_event_key_t key, nanos_event_value_t value, InstrumentationContextData *icd )
{
   /* Recovering a state event in instrumentation context */
   if ( icd == NULL ) icd = myThread->getCurrentWD()->getInstrumentationContextData();

   InstrumentationContextData::BurstIterator it;

   /* find given key in the burst list */
   if ( _instrumentationContext.findBurstByKey( icd, key, it ) ) {
      /* Creating burst event */
      new (e) Event(*it);
      e->reverseType();
      _instrumentationContext.removeBurst( icd, it ); 
   }
   else {
      new (e) Burst( false, key, (nanos_event_value_t) value );
   }

   /* If not needed to show stacked bursts then close current event by openning next one (if any)  */
   if ( ( !_instrumentationContext.showStackedBursts()) && (_instrumentationContext.findBurstByKey( icd, key, it )) ) {
      new (e) Event(*it);
   }
}
void Instrumentation::createPointEvent ( Event *e, nanos_event_key_t key, nanos_event_value_t value )
{
   /* Creating a point event */
   new (e) Point ( key, value );

}

void Instrumentation::createPtPStart ( Event *e, nanos_event_domain_t domain, nanos_event_id_t id,
                                       nanos_event_key_t keys, nanos_event_value_t values, unsigned int partner )
{
   /* Creating a PtP (start) event */
   new (e) PtP ( true, domain, id, keys, values, partner );
}

void Instrumentation::createPtPEnd ( Event *e, nanos_event_domain_t domain, nanos_event_id_t id,
                                     nanos_event_key_t keys, nanos_event_value_t values, unsigned int partner )
{
   /* Creating a PtP (end) event */
   new (e) PtP ( false, domain, id, keys, values, partner );

}
/* ************************************************************************** */
/* ***          C R E A T I N G   D E F E R R E D   E V E N T S           *** */
/* ************************************************************************** */
void Instrumentation::createDeferredPointEvent ( WorkDescriptor &wd, unsigned int nkvs, nanos_event_key_t *keys,
                                      nanos_event_value_t *values )
{
   unsigned int i,ne=0; // Number of events
   Event *e = (Event *) alloca(sizeof(Event) * nkvs ); /* Event array */

   InstrumentationContextData *icd = wd.getInstrumentationContextData();                                             

   /* Create point event */
   for ( i = 0; i < nkvs; i++ ) {
      if (keys[i] != 0 ){
         createPointEvent ( &e[ne], keys[i], values[i] );
         _instrumentationContext.insertDeferredEvent( icd, e[ne++] );
      }
   }

   if ( ne == 0 ) return;

   /* Spawning point events */
   addEventList ( ne, e );
}

void Instrumentation::createDeferredPtPStart ( WorkDescriptor &wd, nanos_event_domain_t domain, nanos_event_id_t id,
                                               nanos_event_key_t key, nanos_event_value_t value, unsigned int partner )
{
   if ( _emitPtPEvents == false ) return;

   Event e; /* Event */

   /* Create event: PtP */
   createPtPStart( &e, domain, id, key, value, partner );

   /* Inserting event into deferred event list */
   InstrumentationContextData *icd = wd.getInstrumentationContextData();                                             
   _instrumentationContext.insertDeferredEvent( icd, e );
}

void Instrumentation::createDeferredPtPEnd ( WorkDescriptor &wd, nanos_event_domain_t domain, nanos_event_id_t id,
                                             nanos_event_key_t key, nanos_event_value_t value, unsigned int partner )
{
   if ( _emitPtPEvents == false ) return;
   Event e; /* Event */

   /* Create event: PtP */
   createPtPEnd( &e, domain, id, key, value, partner );

   /* Inserting event into deferred event list */
   InstrumentationContextData *icd = wd.getInstrumentationContextData();                                             
   _instrumentationContext.insertDeferredEvent( icd, e );
}
/* ************************************************************************** */
/* ***                   T H R O W I N G   E V E N T S                    *** */
/* ************************************************************************** */
void Instrumentation::raisePointEvents ( unsigned int nkvs, nanos_event_key_t *key, nanos_event_value_t *val )
{
   unsigned int i,ne=0; // Number of events
   Event *e = (Event *) alloca(sizeof(Event) * nkvs ); /* Event array */

   /* Create point event */
   for ( i = 0; i < nkvs; i++ )
      if (key[i] != 0 ) createPointEvent ( &e[ne++], key[i], val[i] );

   if ( ne == 0 ) return;

   /* Spawning point events */
   addEventList ( ne, e );
}

void Instrumentation::raiseOpenStateEvent ( nanos_event_state_value_t state )
{
   if ( !_emitStateEvents ) return;
   Event e; /* Event */

   /* Create state event */
   createStateEvent( &e, state );

   /* Spawning state event */
   addEventList ( 1, &e );
}

void Instrumentation::raiseCloseStateEvent ( void )
{
   if ( !_emitStateEvents ) return;
   Event e; /* Event */

   /* Create state event */
   returnPreviousStateEvent( &e );

   /* Spawning state event */
   addEventList ( 1, &e );

}

void Instrumentation::raiseOpenBurstEvent ( nanos_event_key_t key, nanos_event_value_t val )
{
   if ( key == 0 ) return; // key == 0 means disabled event
   Event e; /* Event */

   /* Create event: BURST */
   createBurstEvent( &e, key, val );

   /* Spawning event: specific instrumentation call */
   addEventList ( 1, &e );
}

void Instrumentation::raiseCloseBurstEvent ( nanos_event_key_t key, nanos_event_value_t value )
{
   if ( key == 0 ) return; // key == 0 means disabled event

   Event e; /* Event */

   /* Create event: BURST */
   closeBurstEvent( &e, key, value );

   /* Spawning event: specific instrumentation call */
   addEventList ( 1, &e );
}

void Instrumentation::raiseOpenPtPEvent ( nanos_event_domain_t domain, nanos_event_id_t id, nanos_event_key_t key, nanos_event_value_t val, unsigned int partner )
{
   if ( _emitPtPEvents == false ) return;
   if ( ! ( domain == NANOS_XFER_DATA || domain == NANOS_XFER_REQ || domain == NANOS_XFER_WAIT_REQ_PUT || domain == NANOS_AM_WORK || domain == NANOS_AM_WORK_DONE || domain == NANOS_XFER_FREE_TMP_BUFF || domain == NANOS_WD_REMOTE || domain == NANOS_WAIT || domain == NANOS_WD_DEPENDENCY || domain == NANOS_WD_DOMAIN ) ) return;

   Event e; /* Event */

   /* Create event: PtP */
   createPtPStart( &e, domain, id, key, val, partner );

   /* Spawning event: specific instrumentation call */
   addEventList ( 1, &e );
}
void Instrumentation::raiseClosePtPEvent ( nanos_event_domain_t domain, nanos_event_id_t id, nanos_event_key_t key, nanos_event_value_t val, unsigned int partner )
{
   if ( _emitPtPEvents == false ) return;
   if ( ! ( domain == NANOS_XFER_DATA || domain == NANOS_XFER_REQ || domain == NANOS_XFER_WAIT_REQ_PUT || domain == NANOS_AM_WORK || domain == NANOS_AM_WORK_DONE || domain == NANOS_XFER_FREE_TMP_BUFF || domain == NANOS_WD_REMOTE || domain == NANOS_WAIT || domain == NANOS_WD_DEPENDENCY || domain == NANOS_WD_DOMAIN ) ) return;

   Event e; /* Event */

   /* Create event: PtP */
   createPtPEnd( &e, domain, id, key, val, partner );

   /* Spawning event: specific instrumentation call */
   addEventList ( 1, &e );

}
void Instrumentation::raiseOpenStateAndBurst ( nanos_event_state_value_t state, nanos_event_key_t key, nanos_event_value_t val )
{
   int ne = 0; // Number of max events
   Event e[2]; // Event array

   /* Create state event */
   if ( _emitStateEvents == true ) createStateEvent( &e[ne++], state );

   /* Create burst event */
   if ( key != 0 ) createBurstEvent( &e[ne++], key, val );

   if ( ne == 0 ) return;

   /* Spawning ne events: specific instrumentation call */
   addEventList ( ne, e );

}

void Instrumentation::raiseCloseStateAndBurst ( nanos_event_key_t key, nanos_event_value_t value )
{
   int ne = 0; // Number of max events
   Event e[2]; // Event array

   /* Create state event */
   if ( _emitStateEvents == true ) returnPreviousStateEvent( &e[ne++] );

   /* Create burst event */
   if ( key != 0 ) closeBurstEvent( &e[ne++], key, value );
 
   if ( ne == 0 ) return;

   /* Spawning ne events: specific instrumentation call */
   addEventList ( ne, e );
}

/* ************************************************************************** */
/* ***            C O N T E X T   S W I T C H    E V E N T S              *** */
/* ************************************************************************** */

void Instrumentation::wdCreate( WorkDescriptor* newWD )
{
   Event e1,e2,e3,e4; /* Event */

   /* Gets key for wd-id bursts and wd->id as value*/
   InstrumentationContextData *icd = newWD->getInstrumentationContextData();

   /* Create event: BURST */
   static nanos_event_key_t key = getInstrumentationDictionary()->getEventKey("wd-id");

   if ( key != 0 ) {
      nanos_event_value_t wd_id = newWD->getId();
      createBurstEvent( &e2, key, wd_id, icd );
   }
   
   static nanos_event_key_t priorityKey = getInstrumentationDictionary()->getEventKey("wd-priority");
   nanos_event_value_t wd_priority = (nanos_event_value_t) newWD->getPriority() + 1;
   createBurstEvent( &e3, priorityKey, wd_priority, icd );

   static nanos_event_key_t numaNodeKey = getInstrumentationDictionary()->getEventKey("wd-numa-node");
   nanos_event_value_t wd_numa_node = (nanos_event_value_t) newWD->getNUMANode() + 1;
   createBurstEvent( &e4, numaNodeKey, wd_numa_node, icd );
 
   /* Create event: STATE */
   if ( _emitStateEvents == true ) createStateEvent( &e1, NANOS_RUNTIME, icd );

   /* insert burst as deferred event if context switch is not enabled */
   if ( !_instrumentationContext.isContextSwitchEnabled() ) {
      if ( _emitStateEvents == true ) _instrumentationContext.insertDeferredEvent( icd, e1 );
      if ( key != 0 ) _instrumentationContext.insertDeferredEvent( icd, e2 );
      if ( priorityKey != 0 )_instrumentationContext.insertDeferredEvent( icd, e3 );
      if ( numaNodeKey != 0 ) _instrumentationContext.insertDeferredEvent( icd, e4 );
   }
}

void Instrumentation::flushDeferredEvents ( WorkDescriptor* wd )
{

   if ( !wd ) return;

   InstrumentationContextData *icd = wd->getInstrumentationContextData();
   int numEvents = _instrumentationContext.getNumDeferredEvents ( icd );

   if ( numEvents == 0 ) return;

   Event *e = (Event *) alloca( sizeof( Event ) * numEvents );

   int i = 0;
   InstrumentationContextData::EventIterator itDE;
   for ( itDE  = _instrumentationContext.beginDeferredEvents( icd );
         itDE != _instrumentationContext.endDeferredEvents( icd ); itDE++ ) {
      e[i++] = *itDE;
   }
   _instrumentationContext.clearDeferredEvents( icd );

   addEventList ( numEvents, e );

}

void Instrumentation::wdSwitch( WorkDescriptor* oldWD, WorkDescriptor* newWD, bool last )
{
   unsigned int i = 0;
   unsigned int oldPtP = 0, oldStates = 0, oldBursts = 0;
   unsigned int newPtP = 0, newStates = 0, newBursts = 0, newDeferred = 0;
   InstrumentationContextData *old_icd = NULL;
   InstrumentationContextData *new_icd = NULL;


   /* Computing number of leaving wd related events*/
   if ( oldWD!=NULL ) {
      /* Getting Instrumentation Context and computing number of events */
      old_icd = oldWD->getInstrumentationContextData();
      if ( _emitPtPEvents == true ) oldPtP = last ? 0 : 1;
      oldStates = _instrumentationContext.getNumStates( old_icd );
      oldBursts = _instrumentationContext.getNumBursts( old_icd );
   }

   /* Computing number of entering wd related events*/
   if ( newWD!=NULL ) {
      /* Getting Instrumentation Context and computing number of events*/
      new_icd = newWD->getInstrumentationContextData();
      if ( _emitPtPEvents == true ) newPtP = 1;
      newStates = _instrumentationContext.getNumStates(new_icd);
      newBursts = _instrumentationContext.getNumBursts( new_icd );
      newDeferred = _instrumentationContext.getNumDeferredEvents ( new_icd );
   }

   /* Allocating Events */
   unsigned int numOldEvents = oldPtP + oldStates + oldBursts;
   unsigned int numNewEvents =  newPtP + newStates + newBursts + newDeferred;
   unsigned int numEvents = numOldEvents + numNewEvents;
   bool csEvent = _emitStateEvents && _instrumentationContext.isContextSwitchEnabled() && ( ((newWD!=NULL)&&(oldWD==NULL)) || ((newWD==NULL)&&(oldWD!=NULL)) );

   Event *e = (Event *) alloca(sizeof(Event) * (numEvents + 1) );

   /* Creating leaving wd events */
   if ( old_icd!= NULL ) {
      /* Creating a starting PtP event (if needed) */
      if (!last && _emitPtPEvents ) ASSIGN_EVENT( e[i++] , PtP , (true,  NANOS_WD_DOMAIN, (nanos_event_id_t) oldWD->getId(), 0, 0) );

      ensure0(i == oldPtP, "Final point-to-point events doesn't fit with computed.");

      /* Creating State event's */
      InstrumentationContextData::ConstStateIterator it_s;
      for ( it_s = _instrumentationContext.beginState( old_icd ); it_s != _instrumentationContext.endState( old_icd ); it_s++ ) {
         ASSIGN_EVENT( e[i++] ,  State , (NANOS_STATE_END, *it_s) );
      }
      ensure0(i == oldPtP + oldStates, "Final state events doesn't fit with computed value.");

      if ( csEvent ) {
         ASSIGN_EVENT( e[i++] , State , ( NANOS_STATE_START, NANOS_CONTEXT_SWITCH ) );
         numOldEvents++;
         numEvents++;
      }

      /* Regenerating reverse bursts for old WD */
      InstrumentationContextData::ConstBurstIterator it;
      for ( it = _instrumentationContext.beginBurst( old_icd ); it != _instrumentationContext.endBurst( old_icd ); it++ ) {
         e[i] = *it; e[i++].reverseType();
      }
   }


   /* Creating entering wd events */
   //if ( oldWD != NULL ) {
   //   std::cerr << "Thread " << myThread->getId() << " exitign wd " << oldWD->getId() << " new_icd is " << (void *) new_icd << std::endl;
   //}
   if ( new_icd != NULL) {
      /* Creating PtP event */
      if ( _emitPtPEvents ) ASSIGN_EVENT( e[i++] , PtP , (false, NANOS_WD_DOMAIN, (nanos_event_id_t) newWD->getId(), 0, 0) );

      if ( csEvent ) {
         ASSIGN_EVENT( e[i++] , State , ( NANOS_STATE_END, NANOS_CONTEXT_SWITCH ) );
         numNewEvents++;
         numEvents++;
      }

      /* Creating State event's */
      InstrumentationContextData::ConstStateIterator it_s;
      for ( it_s = _instrumentationContext.beginState( new_icd ); it_s != _instrumentationContext.endState( new_icd ); it_s++) {
         ASSIGN_EVENT( e[i++] , State , ( NANOS_STATE_START, *it_s ) );
      }
      /* Regenerating bursts for new WD: in reverse order */
      InstrumentationContextData::ConstBurstIterator it;
      i += (newBursts-1);
      for ( it = _instrumentationContext.beginBurst( new_icd ) ; it != _instrumentationContext.endBurst( new_icd ); it++ ) {
         e[i--] = *it;
      }
      i += (newBursts+1);

      /* Generating deferred events for new WD (and removing them) */
      InstrumentationContextData::EventIterator itDE;
      for ( itDE  = _instrumentationContext.beginDeferredEvents( new_icd );
            itDE != _instrumentationContext.endDeferredEvents(new_icd); itDE++ ) {
         e[i++] = *itDE;
      }
      _instrumentationContext.clearDeferredEvents( new_icd );

   }

   ensure0( i == numEvents , "Computed number of events doesn't fit with number of real events");

   /* Spawning 'numEvents' events: specific instrumentation call */
   if ( _instrumentationContext.isContextSwitchEnabled() ) {
      if ( numEvents != 0 ) addEventList ( numEvents, &e[0] );
   } else {
      if ( oldWD != NULL) {
         if ( numOldEvents != 0 ) addEventList ( numOldEvents, &e[0] );
         addSuspendTask( *oldWD, last );
      }
      if ( newWD != NULL) {
         addResumeTask( *newWD );
         if ( numNewEvents != 0 ) addEventList (numNewEvents, &e[numOldEvents]);
      }
   }

   /* Calling array event's destructor: cleaning events */
   for ( i = 0; i < numEvents; i++ ) e[i].~Event();
}

Instrumentation::PtP::PtP ( bool start, nanos_event_domain_t domain, nanos_event_id_t id, nanos_event_key_t key,  nanos_event_value_t value, unsigned int partner )
                   : Event ( start ? NANOS_PTP_START : NANOS_PTP_END , key, value, domain, id, partner ) {
                     //if ( sys.getNetwork()->getNodeNum() == 0 ) {
                     //   if ( domain == NANOS_WD_DOMAIN  ) {
                     //      if ( start ) {
                     //         fprintf(stderr, "[%d] >>> Open DOMAIN with id %lld (key: %x value: %llx partner: %d) \n",
                     //            ((myThread!=NULL)?myThread->getId():-1), id, key, value, partner );
                     //      } else {
                     //         fprintf(stderr, "[%d] <<< Close DOMAIN with id %lld (key: %x value: %llx partner: %d) \n",
                     //            ((myThread!=NULL)?myThread->getId():-1), id, key, value, partner );
                     //      }
                     //      //if ( ((myThread!=NULL)?myThread->getId():-1) == 2 ) {
                     //         sys.printBt();
                     //      //}
                     //   }
                     //}
}

std::string InstrumentationDictionary::getSummary()
{
   LockBlock lock( _lock );
   std::ostringstream s;
   s << "================ Instrumentation Summary =================" << std::endl;
   s << "=== Enabled events:";
   int items = 0;
   KeyMapIterator it = _keyMap.begin();
   while ( it != _keyMap.end() ) {
      if ( items == 0 ) { items = 3; s << std::endl; s << "===  | "; }
      if ( it->second->getId() ) { s << it->first << "(" << it->second->getId() << "), "; items--; }
      it++;
   }
   s << std::endl;

   s << "=== Disabled events:";
   items = 0;
   it = _keyMap.begin();
   while ( it != _keyMap.end() ) {
      if ( items == 0 ) { items = 3; s << std::endl; s << "===  | "; }
      if ( it->second->getId() == 0 ) { s << it->first << ", "; items--; }
      it++;
   }
   s << std::endl;
   return s.str();
}

#endif
