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
#include "instrumentor.hpp"

#include "instrumentorcontext.hpp"
#include "system.hpp"
#include "workdescriptor.hpp"
#include <alloca.h>

using namespace nanos;

#ifdef NANOS_INSTRUMENTATION_ENABLED

/* ************************************************************************** */
/* ***                   C R E A T I N G   E V E N T S                    *** */
/* ************************************************************************** */

void Instrumentation::createStateEvent( Event *e, nanos_event_state_value_t state )
{
   /* Registering a state event in instrucmentor context */
   InstrumentationContextData *icd = myThread->getCurrentWD()->getInstrumentorContextData();
   _instrumentationContext->pushState(icd, state);

   /* Creating a state event */
   if ( _instrumentationContext->isStateEventEnabled( icd ) ) new (e) State(STATE, state);
   else new (e) State(SUBSTATE, state);
}

void Instrumentation::returnPreviousStateEvent ( Event *e )
{
   /* Recovering a state event in instrumentor context */
   InstrumentationContextData  *icd = myThread->getCurrentWD()->getInstrumentorContextData();
   _instrumentationContext->popState( icd );
   nanos_event_state_value_t state = _instrumentationContext->topState( icd ); 

   /* Creating a state event */
   if ( _instrumentationContext->isStateEventEnabled( icd ) ) new (e) State(STATE,state);
   else new (e) State(SUBSTATE, state);
}

void Instrumentation::createBurstEvent ( Event *e, nanos_event_key_t key, nanos_event_value_t value )
{
   /* Creating burst  event */
   Event::KV kv( key, value );
   new (e) Burst( true, kv );

   InstrumentationContextData *icd = myThread->getCurrentWD()->getInstrumentorContextData();
   _instrumentationContext->insertBurst( icd, *e );
}

void Instrumentation::closeBurstEvent ( Event *e, nanos_event_key_t key )
{
   /* Removing burst event in instrumentation context */
   InstrumentationContextData *icd = myThread->getCurrentWD()->getInstrumentorContextData();
   InstrumentationContextData::BurstIterator it;

   /* find given key in the burst list */
   if ( _instrumentationContext->findBurstByKey( icd, key, it ) ) {
      /* Creating burst event */
      new (e) Event(*it);
      e->reverseType();
      _instrumentationContext->removeBurst( icd, it ); 
   }
   else fatal("Burst type doesn't exists");

   /* If not needed to show stacked bursts then close current event by openning next one (if any)  */
   if ( ( !_instrumentationContext->showStackedBursts()) && (_instrumentationContext->findBurstByKey( icd, key, it )) ) {
      new (e) Event(*it);
   }
}

void Instrumentation::createPointEvent ( Event *e, unsigned int nkvs, nanos_event_key_t *keys,
                                      nanos_event_value_t *values )
{
   /* Creating an Event::KV vector */
   Event::KVList kvlist = new Event::KV[nkvs];

   /* Initializing kvlist elements */
   for ( unsigned int i = 0; i < nkvs; i++ ) {
      kvlist[i] = Event::KV ( keys[i], values[i] );
   }

   /* Creating a point event */
   new (e) Point ( nkvs, kvlist );

}

void Instrumentation::createPtPStart ( Event *e, nanos_event_domain_t domain, nanos_event_id_t id,
                      unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values )
{
   /* Creating an Event::KV vector */
   Event::KVList kvlist = new Event::KV[nkvs];

   /* Initializing kvlist elements */
   for ( unsigned int i = 0; i < nkvs; i++ ) {
      kvlist[i] = Event::KV ( keys[i], values[i] );
   }

   /* Creating a PtP (start) event */
   new (e) PtP ( true, domain, id, nkvs, kvlist );
}

void Instrumentation::createPtPEnd ( Event *e, nanos_event_domain_t domain, nanos_event_id_t id,
                      unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values )
{
   /* Creating an Event::KV vector */
   Event::KVList kvlist = new Event::KV[nkvs];

   /* Initializing kvlist elements */
   for ( unsigned int i = 0; i < nkvs; i++ ) {
      kvlist[i] = Event::KV ( keys[i], values[i] );
   }

   /* Creating a PtP (end) event */
   new (e) PtP ( false, domain, id, nkvs, kvlist );

}

/* ************************************************************************** */
/* ***                   T H R O W I N G   E V E N T S                    *** */
/* ************************************************************************** */

void Instrumentation::raisePointEvent ( nanos_event_key_t key, nanos_event_value_t val )
{
   Event e; /* Event */

   /* Create point event */
   createPointEvent ( &e, 1, &key, &val );

   /* Spawning point event */
   addEventList ( 1, &e );
}

void Instrumentation::raisePointEventNkvs ( unsigned int nkvs, nanos_event_key_t *key, nanos_event_value_t *val )
{
   Event e; /* Event */

   /* Create point event */
   createPointEvent ( &e, nkvs, key, val );

   /* Spawning point event */
   addEventList ( 1, &e );
}

void Instrumentation::raiseOpenStateEvent ( nanos_event_state_value_t state )
{
   Event e; /* Event */

   /* Create state event */
   createStateEvent( &e, state );

   /* Spawning state event */
   addEventList ( 1, &e );
}

void Instrumentation::raiseCloseStateEvent ( void )
{
   Event e; /* Event */

   /* Create state event */
   returnPreviousStateEvent( &e );

   /* Spawning state event */
   addEventList ( 1, &e );

}

void Instrumentation::raiseOpenBurstEvent ( nanos_event_key_t key, nanos_event_value_t val )
{
   Event e; /* Event */

   /* Create event: BURST */
   createBurstEvent( &e, key, val );

   /* Spawning event: specific instrumentor call */
   addEventList ( 1, &e );
}

void Instrumentation::raiseCloseBurstEvent ( nanos_event_key_t key )
{
   Event e; /* Event */

   /* Create event: BURST */
   closeBurstEvent( &e, key );

   /* Spawning event: specific instrumentor call */
   addEventList ( 1, &e );
}

void Instrumentation::raiseOpenPtPEvent ( nanos_event_domain_t domain, nanos_event_id_t id, nanos_event_key_t key, nanos_event_value_t val )
{
   Event e; /* Event */

   /* Create event: PtP */
   createPtPStart( &e, domain, id, 1, &key, &val );

   /* Spawning event: specific instrumentor call */
   addEventList ( 1, &e );
}

void Instrumentation::raiseOpenPtPEventNkvs ( nanos_event_domain_t domain, nanos_event_id_t id, unsigned int nkvs,
                                           nanos_event_key_t *key, nanos_event_value_t *val )
{
   Event e; /* Event */

   /* Create event: PtP */
   createPtPStart( &e, domain, id, nkvs, key, val );

   /* Spawning event: specific instrumentor call */
   addEventList ( 1, &e );
}

void Instrumentation::raiseClosePtPEvent ( nanos_event_domain_t domain, nanos_event_id_t id, nanos_event_key_t key, nanos_event_value_t val )
{
   Event e; /* Event */

   /* Create event: PtP */
   createPtPEnd( &e, domain, id, 1, &key, &val );

   /* Spawning event: specific instrumentor call */
   addEventList ( 1, &e );

}

void Instrumentation::raiseClosePtPEventNkvs ( nanos_event_domain_t domain, nanos_event_id_t id, unsigned int nkvs,
                                            nanos_event_key_t *key, nanos_event_value_t *val )
{
   Event e; /* Event */

   /* Create event: PtP */
   createPtPEnd( &e, domain, id, nkvs, key, val );

   /* Spawning event: specific instrumentor call */
   addEventList ( 1, &e );

}

void Instrumentation::raiseOpenStateAndBurst ( nanos_event_state_value_t state, nanos_event_key_t key, nanos_event_value_t val )
{
   Event e[2]; /* Event array */

   /* Create a vector of two events: STATE and BURST */
   createStateEvent( &e[0], state );
   createBurstEvent( &e[1], key, val );

   /* Spawning two events: specific instrumentor call */
   addEventList ( 2, e );

}

void Instrumentation::raiseCloseStateAndBurst ( nanos_event_key_t key )
{
   Event e[2]; /* Event array */

   /* Creating a vector of two events: STATE and BURST */
   returnPreviousStateEvent( &e[0] );
   closeBurstEvent( &e[1], key ); 
 
   /* Spawning two events: specific instrumentor call */
   addEventList ( 2, e );
 
}

/* ************************************************************************** */
/* ***            C O N T E X T   S W I T C H    E V E N T S              *** */
/* ************************************************************************** */

void Instrumentation::wdCreate( WorkDescriptor* newWD )
{
   /* Gets key for wd-id bursts and wd->id as value*/
   static nanos_event_key_t key = getInstrumentorDictionary()->getEventKey("wd-id");
   nanos_event_value_t wd_id = newWD->getId();

   /* Creating key value and Burst event */
   Event::KV kv( key, wd_id );
   Event *e = new Burst( true, kv );

   InstrumentationContextData *icd = newWD->getInstrumentorContextData();
 
   _instrumentationContext->insertBurst( icd, *e );
   _instrumentationContext->pushState( icd, RUNTIME );
}


void Instrumentation::wdEnterCPU( WorkDescriptor* newWD )
{
   unsigned int i = 0; /* Used as Event e[] index */

   /* Computing number of burst events */
   InstrumentationContextData *icd = newWD->getInstrumentorContextData();
   unsigned int newBursts = _instrumentationContext->getNumBursts( icd );
   unsigned int numEvents = 2 + newBursts;
   if ( !_instrumentationContext->isStateEventEnabled( icd ) ) numEvents++;

   /* Allocating Events */
   Event *e = (Event *) alloca(sizeof(Event) * numEvents );

   /* Creating two PtP events */
   e[i++] = PtP (false, NANOS_WD_DOMAIN, (nanos_event_id_t) newWD->getId(), 0, NULL);

   /* Creating State event: change thread current state with newWD saved state */
   nanos_event_state_value_t state;

   if ( _instrumentationContext->isStateEventEnabled( icd ) )
      state = _instrumentationContext->topState( icd );
   else {
      state = _instrumentationContext->topState( icd );
      e[i++] = State ( SUBSTATE, state );
      state = _instrumentationContext->getValidState( icd );
   }

   e[i++] = State ( STATE, state );

   /* Regenerating bursts for new WD */
   i += (newBursts-1);
   for ( InstrumentationContextData::ConstBurstIterator it = _instrumentationContext->beginBurst( icd ) ; it != _instrumentationContext->endBurst( icd ); it++,i-- ) {
      e[i] = *it;
   }
   i += (newBursts);

   /* Spawning 'numEvents' events: specific instrumentor call */
   addEventList ( numEvents, e );
}

void Instrumentation::wdLeaveCPU( WorkDescriptor* oldWD )
{
   unsigned int i = 0; /* Used as Event e[] index */

   /* Computing number of burst events */
   InstrumentationContextData *icd = oldWD->getInstrumentorContextData();
   unsigned int oldBursts = _instrumentationContext->getNumBursts( icd );
   unsigned int numEvents = 2 + oldBursts;
   if ( !_instrumentationContext->isStateEventEnabled( icd ) ) numEvents++;

   /* Allocating Events */
   Event *e = (Event *) alloca(sizeof(Event) * numEvents );

   /* Creating two PtP events */
   e[i++] = PtP (true,  NANOS_WD_DOMAIN, (nanos_event_id_t) oldWD->getId(), 0, NULL);

   /* Creating State event: change thread current state with newWD saved state */
   if ( !_instrumentationContext->isStateEventEnabled( icd ) ) e[i++] = State ( SUBSTATE, NOT_TRACED );
   e[i++] = State ( STATE, RUNTIME );

   /* Regenerating reverse bursts for old WD */
   for ( InstrumentationContextData::ConstBurstIterator it = _instrumentationContext->beginBurst( icd ); it != _instrumentationContext->endBurst( icd ); it++,i++ ) {
      e[i] = *it;
      e[i].reverseType();
   }

   /* Spawning 'numEvents' events: specific instrumentor call */
   addEventList ( numEvents, e );
}

void Instrumentation::wdExit( WorkDescriptor* oldWD, WorkDescriptor* newWD )
{
   unsigned int i = 0; /* Used as Event e[] index */

   /* Computing number of events */
   InstrumentationContextData *old_icd = oldWD->getInstrumentorContextData();
   InstrumentationContextData *new_icd = newWD->getInstrumentorContextData();
   unsigned int oldBursts = _instrumentationContext->getNumBursts( old_icd );
   unsigned int newBursts = _instrumentationContext->getNumBursts( new_icd );
   unsigned int numEvents = 2 + oldBursts + newBursts;
   if ( !_instrumentationContext->isStateEventEnabled( new_icd ) ) numEvents++;
   if ( !_instrumentationContext->isStateEventEnabled( old_icd ) ) numEvents++;

   /* Allocating Events */
   Event *e = (Event *) alloca(sizeof(Event) * numEvents );

   /* Creating PtP event: as oldWD has finished execution we need to generate only PtP End
    * in order to instrument receiving point for the new WorkDescriptor */
   e[i++] = PtP ( false, NANOS_WD_DOMAIN, (nanos_event_id_t) newWD->getId(), 0, NULL );

   /* Creating State event: change thread current state with newWD saved state */
   nanos_event_state_value_t state;

   if ( !_instrumentationContext->isStateEventEnabled( old_icd ) ) e[i++] = State ( SUBSTATE, NOT_TRACED );

   if ( _instrumentationContext->isStateEventEnabled( new_icd ) )
      state = _instrumentationContext->topState( new_icd );
   else {
      state = _instrumentationContext->topState( new_icd );
      e[i++] = State ( SUBSTATE, state );
      state = _instrumentationContext->getValidState( new_icd );
   }
   e[i++] = State ( STATE, state );

   /* Regenerating reverse bursts for old WD */
   for ( InstrumentationContextData::ConstBurstIterator it = _instrumentationContext->beginBurst( old_icd) ; it != _instrumentationContext->endBurst( old_icd ); it++,i++) {
      e[i] = *it;
      e[i].reverseType();
   }

   /* Regenerating bursts for new WD */
   i += (newBursts-1);
   for ( InstrumentationContextData::ConstBurstIterator it = _instrumentationContext->beginBurst( new_icd ) ; it != _instrumentationContext->endBurst( new_icd); it++,i--) {
      e[i] = *it;
   }
   i += newBursts;

   /* Spawning 'numEvents' events: specific instrumentor call */
   addEventList ( numEvents, e );
}

void Instrumentation::enableStateEvents()
{
   InstrumentationContextData *icd = myThread->getCurrentWD()->getInstrumentorContextData();
   _instrumentationContext->enableStateEvents( icd );
   Event e = State ( SUBSTATE, NOT_TRACED );
   addEventList ( 1, &e );
}

void Instrumentation::disableStateEvents()
{
   InstrumentationContextData *icd = myThread->getCurrentWD()->getInstrumentorContextData();
   _instrumentationContext->disableStateEvents( icd );
   _instrumentationContext->setValidState( icd, _instrumentationContext->topState( icd ) );
}

#endif
