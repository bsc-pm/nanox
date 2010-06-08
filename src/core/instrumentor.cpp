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

/* ************************************************************************** */
/* ***                   C R E A T I N G   E V E N T S                    *** */
/* ************************************************************************** */

void Instrumentor::createStateEvent( Event *e, nanos_event_state_value_t state )
{
   /* Registering a state event in instrucmentor context */
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.pushState(state);

   /* Creating a state event */
   if ( instrContext.isStateEventEnabled() ) new (e) State(STATE, state);
   else new (e) State(SUBSTATE, state);
}

void Instrumentor::returnPreviousStateEvent ( Event *e )
{
   /* Recovering a state event in instrumentor context */
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.popState();
   nanos_event_state_value_t state = instrContext.topState(); 

   /* Creating a state event */
   if ( instrContext.isStateEventEnabled() ) new (e) State(STATE,state);
   else new (e) State(SUBSTATE, state);
}

void Instrumentor::createBurstEvent ( Event *e, nanos_event_key_t key, nanos_event_value_t value )
{
   /* Creating burst  event */
   Event::KV kv( key, value );
   new (e) Burst( true, kv );

   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.insertBurst( *e );
}

void Instrumentor::closeBurstEvent ( Event *e, nanos_event_key_t key )
{
   /* Removing burst event in instrucmentor context */
   InstrumentorContext &ic = myThread->getCurrentWD()->getInstrumentorContext();
   InstrumentorContext::BurstIterator it;
   if ( ic.findBurstByKey( key, it ) ) {
      /* Creating burst event */
      new (e) Event(*it);
      e->reverseType();
      ic.removeBurst( it ); 
   }
   else fatal("Burst type doesn't exists");
}

void Instrumentor::createPointEvent ( Event *e, unsigned int nkvs, nanos_event_key_t *keys,
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

void Instrumentor::createPtPStart ( Event *e, nanos_event_domain_t domain, nanos_event_id_t id,
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

void Instrumentor::createPtPEnd ( Event *e, nanos_event_domain_t domain, nanos_event_id_t id,
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

void Instrumentor::throwPointEvent ( nanos_event_key_t key, nanos_event_value_t val )
{
   Event e; /* Event */

   /* Create point event */
   createPointEvent ( &e, 1, &key, &val );

   /* Spawning point event */
   addEventList ( 1, &e );
}

void Instrumentor::throwPointEventNkvs ( unsigned int nkvs, nanos_event_key_t *key, nanos_event_value_t *val )
{
   Event e; /* Event */

   /* Create point event */
   createPointEvent ( &e, nkvs, key, val );

   /* Spawning point event */
   addEventList ( 1, &e );
}

void Instrumentor::throwOpenStateEvent ( nanos_event_state_value_t state )
{
   Event e; /* Event */

   /* Create state event */
   createStateEvent( &e, state );

   /* Spawning state event */
   addEventList ( 1, &e );
}

void Instrumentor::throwCloseStateEvent ( void )
{
   Event e; /* Event */

   /* Create state event */
   returnPreviousStateEvent( &e );

   /* Spawning state event */
   addEventList ( 1, &e );

}

void Instrumentor::throwOpenBurstEvent ( nanos_event_key_t key, nanos_event_value_t val )
{
   Event e; /* Event */

   /* Create event: BURST */
   createBurstEvent( &e, key, val );

   /* Spawning event: specific instrumentor call */
   addEventList ( 1, &e );
}

void Instrumentor::throwCloseBurstEvent ( nanos_event_key_t key )
{
   Event e; /* Event */

   /* Create event: BURST */
   closeBurstEvent( &e, key );

   /* Spawning event: specific instrumentor call */
   addEventList ( 1, &e );
}

void Instrumentor::throwOpenPtPEvent ( nanos_event_domain_t domain, nanos_event_id_t id, nanos_event_key_t key, nanos_event_value_t val )
{
   Event e; /* Event */

   /* Create event: PtP */
   createPtPStart( &e, domain, id, 1, &key, &val );

   /* Spawning event: specific instrumentor call */
   addEventList ( 1, &e );
}

void Instrumentor::throwOpenPtPEventNkvs ( nanos_event_domain_t domain, nanos_event_id_t id, unsigned int nkvs,
                                           nanos_event_key_t *key, nanos_event_value_t *val )
{
   Event e; /* Event */

   /* Create event: PtP */
   createPtPStart( &e, domain, id, nkvs, key, val );

   /* Spawning event: specific instrumentor call */
   addEventList ( 1, &e );
}

void Instrumentor::throwClosePtPEvent ( nanos_event_domain_t domain, nanos_event_id_t id, nanos_event_key_t key, nanos_event_value_t val )
{
   Event e; /* Event */

   /* Create event: PtP */
   createPtPEnd( &e, domain, id, 1, &key, &val );

   /* Spawning event: specific instrumentor call */
   addEventList ( 1, &e );

}

void Instrumentor::throwClosePtPEventNkvs ( nanos_event_domain_t domain, nanos_event_id_t id, unsigned int nkvs,
                                            nanos_event_key_t *key, nanos_event_value_t *val )
{
   Event e; /* Event */

   /* Create event: PtP */
   createPtPEnd( &e, domain, id, nkvs, key, val );

   /* Spawning event: specific instrumentor call */
   addEventList ( 1, &e );

}

void Instrumentor::throwOpenStateAndBurst ( nanos_event_state_value_t state, nanos_event_key_t key, nanos_event_value_t val )
{
   Event e[2]; /* Event array */

   /* Create a vector of two events: STATE and BURST */
   createStateEvent( &e[0], state );
   createBurstEvent( &e[1], key, val );

   /* Spawning two events: specific instrumentor call */
   addEventList ( 2, e );

}

void Instrumentor::throwCloseStateAndBurst ( nanos_event_key_t key )
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

void Instrumentor::wdCreate( WorkDescriptor* newWD )
{
   /* Gets key for wd-id bursts and wd->id as value*/
   static nanos_event_key_t key = getInstrumentorDictionary()->getEventKey("wd-id");
   nanos_event_value_t wd_id = newWD->getId();

   /* Creating key value and Burst event */
   Event::KV kv( key, wd_id );
   Event *e = new Burst( true, kv );

   InstrumentorContext &instrContext = newWD->getInstrumentorContext();
 
   instrContext.insertBurst( *e );
   instrContext.pushState( RUNTIME );
}


void Instrumentor::wdEnterCPU( WorkDescriptor* newWD )
{
   unsigned int i = 0; /* Used as Event e[] index */

   /* Computing number of burst events */
   InstrumentorContext &newInstrContext = newWD->getInstrumentorContext();
   unsigned int newBursts = newInstrContext.getNumBursts();
   unsigned int numEvents = 2 + newBursts;
   if ( !newInstrContext.isStateEventEnabled() ) numEvents++;

   /* Allocating Events */
   Event *e = (Event *) alloca(sizeof(Event) * numEvents );

   /* Creating two PtP events */
   e[i++] = PtP (false, NANOS_WD_DOMAIN, (nanos_event_id_t) newWD->getId(), 0, NULL);

   /* Creating State event: change thread current state with newWD saved state */
   nanos_event_state_value_t state;

   if ( newInstrContext.isStateEventEnabled() )
      state = newInstrContext.topState();
   else {
      state = newInstrContext.topState();
      e[i++] = State ( SUBSTATE, state );
      state = newInstrContext.validState();
   }

   e[i++] = State ( STATE, state );

   /* Regenerating bursts for new WD */
   for ( InstrumentorContext::ConstBurstIterator it = newInstrContext.beginBurst() ; it != newInstrContext.endBurst(); it++,i++ ) {
      e[i] = *it;
   }

   /* Spawning 'numEvents' events: specific instrumentor call */
   addEventList ( numEvents, e );
}

void Instrumentor::wdLeaveCPU( WorkDescriptor* oldWD )
{
   unsigned int i = 0; /* Used as Event e[] index */

   /* Computing number of burst events */
   InstrumentorContext &oldInstrContext = oldWD->getInstrumentorContext();
   unsigned int oldBursts = oldInstrContext.getNumBursts();
   unsigned int numEvents = 2 + oldBursts;
   if ( !oldInstrContext.isStateEventEnabled() ) numEvents++;

   /* Allocating Events */
   Event *e = (Event *) alloca(sizeof(Event) * numEvents );

   /* Creating two PtP events */
   e[i++] = PtP (true,  NANOS_WD_DOMAIN, (nanos_event_id_t) oldWD->getId(), 0, NULL);

   /* Creating State event: change thread current state with newWD saved state */
   if ( !oldInstrContext.isStateEventEnabled() ) e[i++] = State ( SUBSTATE, NOT_TRACED );
   e[i++] = State ( STATE, RUNTIME );

   /* Regenerating reverse bursts for old WD */
   for ( InstrumentorContext::ConstBurstIterator it = oldInstrContext.beginBurst(); it != oldInstrContext.endBurst(); it++,i++ ) {
      e[i] = *it;
      e[i].reverseType();
   }

   /* Spawning 'numEvents' events: specific instrumentor call */
   addEventList ( numEvents, e );
}

void Instrumentor::wdExit( WorkDescriptor* oldWD, WorkDescriptor* newWD )
{
   unsigned int i = 0; /* Used as Event e[] index */

   /* Computing number of events */
   InstrumentorContext &oldInstrContext = oldWD->getInstrumentorContext();
   InstrumentorContext &newInstrContext = newWD->getInstrumentorContext();
   unsigned int oldBursts = oldInstrContext.getNumBursts();
   unsigned int newBursts = newInstrContext.getNumBursts();
   unsigned int numEvents = 2 + oldBursts + newBursts;
   if ( !newInstrContext.isStateEventEnabled() ) numEvents++;
   if ( !oldInstrContext.isStateEventEnabled() ) numEvents++;

   /* Allocating Events */
   Event *e = (Event *) alloca(sizeof(Event) * numEvents );

   /* Creating PtP event: as oldWD has finished execution we need to generate only PtP End
    * in order to instrument receiving point for the new WorkDescriptor */
   e[i++] = PtP ( false, NANOS_WD_DOMAIN, (nanos_event_id_t) newWD->getId(), 0, NULL );

   /* Creating State event: change thread current state with newWD saved state */
   nanos_event_state_value_t state;

   if ( !oldInstrContext.isStateEventEnabled() ) e[i++] = State ( SUBSTATE, NOT_TRACED );

   if ( newInstrContext.isStateEventEnabled() )
      state = newInstrContext.topState();
   else {
      state = newInstrContext.topState();
      e[i++] = State ( SUBSTATE, state );
      state = newInstrContext.validState();
   }
   e[i++] = State ( STATE, state );

   /* Regenerating reverse bursts for old WD */
   for ( InstrumentorContext::ConstBurstIterator it = oldInstrContext.beginBurst() ; it != oldInstrContext.endBurst(); it++,i++ ) {
      e[i] = *it;
      e[i].reverseType();
   }

   /* Regenerating bursts for new WD */
   for ( InstrumentorContext::ConstBurstIterator it = newInstrContext.beginBurst() ; it != newInstrContext.endBurst(); it++,i++ ) {
      e[i] = *it;
   }

   /* Spawning 'numEvents' events: specific instrumentor call */
   addEventList ( numEvents, e );
}

void Instrumentor::enableStateEvents()
{
   InstrumentorContext &ic = myThread->getCurrentWD()->getInstrumentorContext();
   ic.enableStateEvents();
   Event e = State ( SUBSTATE, NOT_TRACED );
   addEventList ( 1, &e );
}

void Instrumentor::disableStateEvents()
{
   InstrumentorContext &ic = myThread->getCurrentWD()->getInstrumentorContext();
   ic.disableStateEvents();
   ic.saveValidState();
}

/* ************************************************************************** */
/* ***      D E P R E C A T E D    F U N C T I O N    E V E N T S         *** */
/* ************************************************************************** */

void Instrumentor::enterRuntimeAPI ( nanos_event_value_t val, nanos_event_state_value_t state )
{
   /* Gets key for api functions */
   static nanos_event_key_t key = getInstrumentorDictionary()->getEventKey("api");
   throwOpenStateAndBurst ( state, key, val );
}

void Instrumentor::leaveRuntimeAPI ( )
{
   /* Gets key for api functions */
   static nanos_event_key_t key = getInstrumentorDictionary()->getEventKey("api");
   throwCloseStateAndBurst ( key );
}

void Instrumentor::registerCopy( nanos_event_key_t key, size_t size )
{
   throwPointEvent ( key, (nanos_event_value_t) size );
}
void Instrumentor::registerCacheHit( nanos_event_key_t key, uint64_t addr )
{
   throwPointEvent ( key, (nanos_event_value_t) addr );
}

void Instrumentor::enterCache( nanos_event_key_t key, size_t size )
{
   throwOpenStateAndBurst ( CACHE, key, (nanos_event_value_t) size );
}

void Instrumentor::leaveCache( nanos_event_key_t key )
{
   throwCloseStateAndBurst( key );
}

void Instrumentor::enterTransfer( nanos_event_key_t key, size_t size )
{
   throwOpenStateAndBurst ( MEM_TRANSFER, key, (nanos_event_value_t) size );
}
void Instrumentor::leaveTransfer( nanos_event_key_t key )
{
   throwCloseStateAndBurst( key );
}
void Instrumentor::enterUserCode ( void )
{
   static nanos_event_key_t key = getInstrumentorDictionary()->getEventKey("user-code");
   nanos_event_value_t val = myThread->getCurrentWD()->getId();
   throwOpenStateAndBurst ( RUNNING, key, val );
}
void Instrumentor::leaveUserCode ( void )
{
   /* Get key for user-code */
   static nanos_event_key_t key = getInstrumentorDictionary()->getEventKey("user-code");
   throwCloseStateAndBurst( key );
}
void Instrumentor::enterIdle ( ) { throwOpenStateEvent ( IDLE ); }
void Instrumentor::leaveIdle ( ) { throwCloseStateEvent (); }
void Instrumentor::enterStartUp ( void ) { throwOpenStateEvent ( STARTUP ); }
void Instrumentor::leaveStartUp ( void ) { throwCloseStateEvent (); }
void Instrumentor::enterShutDown ( void ) { throwOpenStateEvent ( SHUTDOWN ); }
void Instrumentor::leaveShutDown ( void ) { throwCloseStateEvent (); }


