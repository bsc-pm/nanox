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

using namespace nanos;

void Instrumentor::enterRuntimeAPI ( std::string function, std::string description, nanos_event_state_value_t state )
{
   /* Register (if not) key and values */
   InstrumentorDictionary *iD = sys.getInstrumentorDictionary();
   nanos_event_key_t   key = iD->registerEventKey("api","Nanos Runtime API");
   nanos_event_value_t val = iD->registerEventValue( "api", function, description);

   /* Create a vector of two events: STATE and BURST */
   Event::KV kv( Event::KV( key, val ) );
   Event e[2] = { State(state), Burst( true, kv) };

   /* Update instrumentor context with new state and open burst */
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.pushState(state);
   instrContext.insertBurst( e[1] );

   /* Spawning two events: specific instrumentor call */
   addEventList ( 2, e );
}

void Instrumentor::leaveRuntimeAPI ( )
{
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   InstrumentorContext::BurstIterator it;

   InstrumentorDictionary *iD = sys.getInstrumentorDictionary();
   nanos_event_key_t key = iD->registerEventKey("api","Nanos Runtime API");
   if ( !instrContext.findBurstByKey( key, it ) )
      fatal0("Burst doesn't exists");

   Event &e1 =  (*it);
   e1.reverseType();

   /* Top is current state, so before we have to bring (pop) previous state
    * on top of the stack and then restore previous state */
   instrContext.popState();
   nanos_event_state_value_t state = instrContext.topState();

   /* Creating two events */
   Event e[2] = { State(state), e1 };

   /* Spawning two events: specific instrumentor call */
   addEventList ( 2, e );

   instrContext.removeBurst( it ); 
}

void Instrumentor::wdCreate( WorkDescriptor* newWD )
{
   /* Register (if not) key and values */
   InstrumentorDictionary *iD = sys.getInstrumentorDictionary();
   nanos_event_key_t   key = iD->registerEventKey("wd-id","Work Descriptor id:");

   /* Getting work descriptor id */
   nanos_event_value_t wd_id = newWD->getId();

   /* Creating key value and Burst event */
   Event::KV kv( Event::KV( key, wd_id ) );
   Event e = Burst( true, kv );

   InstrumentorContext &instrContext = newWD->getInstrumentorContext();
 
   instrContext.insertBurst( e );
   instrContext.pushState( RUNNING );
}

void Instrumentor::wdSwitchEnter( WorkDescriptor* newWD )
{
   unsigned int i = 0; /* Used as Event e[] index */

   /* Computing number of burst events */
   InstrumentorContext &newInstrContext = newWD->getInstrumentorContext();
   unsigned int newBursts = newInstrContext.getNumBursts();
   unsigned int numEvents = 2 + newBursts;

   /* Allocating Events */
   Event *e = (Event *) alloca(sizeof(Event) * numEvents );

   /* Creating two PtP events */
   e[i++] = PtP (false, NANOS_WD_DOMAIN, (nanos_event_id_t) newWD->getId(), 0, NULL);

   /* Creating State event: change thread current state with newWD saved state */
   nanos_event_state_value_t state = newInstrContext.topState();
   e[i++] = State ( state );

   /* Regenerating bursts for new WD */
   for ( InstrumentorContext::ConstBurstIterator it = newInstrContext.beginBurst() ; it != newInstrContext.endBurst(); it++,i++ ) {
      e[i] = *it;
   }

   /* Spawning 'numEvents' events: specific instrumentor call */
   addEventList ( numEvents, e );
}

void Instrumentor::wdSwitchLeave( WorkDescriptor* oldWD )
{
   unsigned int i = 0; /* Used as Event e[] index */

   /* Computing number of burst events */
   InstrumentorContext &oldInstrContext = oldWD->getInstrumentorContext();
   unsigned int oldBursts = oldInstrContext.getNumBursts();
   unsigned int numEvents = 2 + oldBursts;

   /* Allocating Events */
   Event *e = (Event *) alloca(sizeof(Event) * numEvents );

   /* Creating two PtP events */
   e[i++] = PtP (true,  NANOS_WD_DOMAIN, (nanos_event_id_t) oldWD->getId(), 0, NULL);

   /* Creating State event: change thread current state with newWD saved state */
   e[i++] = State ( RUNTIME );

   /* Regenerating reverse bursts for old WD */
   for ( InstrumentorContext::ConstBurstIterator it = oldInstrContext.beginBurst(); it != oldInstrContext.endBurst(); it++,i++ ) {
      e[i] = *it;
      e[i].reverseType();
   }

   /* Spawning 'numEvents' events: specific instrumentor call */
   addEventList ( numEvents, e );
}

void Instrumentor::wdSwitch( WorkDescriptor* oldWD, WorkDescriptor* newWD )
{
   unsigned int i = 0; /* Used as Event e[] index */

   /* Computing number of burst events */
   InstrumentorContext &oldInstrContext = oldWD->getInstrumentorContext();
   InstrumentorContext &newInstrContext = newWD->getInstrumentorContext();
   unsigned int oldBursts = oldInstrContext.getNumBursts();
   unsigned int newBursts = newInstrContext.getNumBursts();
   unsigned int numEvents = 3 + oldBursts + newBursts;

   /* Allocating Events */
   Event *e = (Event *) alloca(sizeof(Event) * numEvents );

   /* Creating two PtP events */
   e[i++] = PtP (true,  NANOS_WD_DOMAIN, (nanos_event_id_t) oldWD->getId(), 0, NULL);
   e[i++] = PtP (false, NANOS_WD_DOMAIN, (nanos_event_id_t) newWD->getId(), 0, NULL);

   /* Creating State event: change thread current state with newWD saved state */
   nanos_event_state_value_t state = newInstrContext.topState();
   e[i++] = State ( state );

   /* Regenerating reverse bursts for old WD */
   for ( InstrumentorContext::ConstBurstIterator it = oldInstrContext.beginBurst(); it != oldInstrContext.endBurst(); it++,i++ ) {
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

void Instrumentor::wdExit( WorkDescriptor* oldWD, WorkDescriptor* newWD )
{
   unsigned int i = 0; /* Used as Event e[] index */

   /* Computing number of events */
   InstrumentorContext &oldInstrContext = oldWD->getInstrumentorContext();
   InstrumentorContext &newInstrContext = newWD->getInstrumentorContext();
   unsigned int oldBursts = oldInstrContext.getNumBursts();
   unsigned int newBursts = newInstrContext.getNumBursts();
   unsigned int numEvents = 2 + oldBursts + newBursts;

   /* Allocating Events */
   Event *e = (Event *) alloca(sizeof(Event) * numEvents );

   /* Creating PtP event: as oldWD has finished execution we need to generate only PtP End
    * in order to instrument receiving point for the new WorkDescriptor */
   e[i++] = PtP ( false, NANOS_WD_DOMAIN, (nanos_event_id_t) newWD->getId(), 0, NULL );

   /* Creating State event: change thread current state with newWD saved state */
   nanos_event_state_value_t state = newInstrContext.topState();
   e[i++] = State ( state );

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

void Instrumentor::enterIdle ( )
{
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.pushState(IDLE);

   Event e = State(IDLE);
   addEventList ( 1u, &e );
}

void Instrumentor::leaveIdle ( )
{
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.popState();
   nanos_event_state_value_t state = instrContext.topState();

   Event e = State( state );
   addEventList ( 1u, &e );
}

void Instrumentor::createBurstStart ( Event &e, nanos_event_key_t key, nanos_event_value_t value )
{
   /* Creating burst  event */
   Event::KV kv( Event::KV( key, value) );
   e = Burst( true, kv );

   /* Registering burst event in instrucmentor context */
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.insertBurst( e );
}

void Instrumentor::createBurstEnd ( Event &e, nanos_event_key_t key, nanos_event_value_t value )
{
   /* Creating burst event */
   Event::KV kv( Event::KV( key, value) );
   e = Burst( false, kv );

   /* Deleting burst event in instrucmentor context */
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   InstrumentorContext::BurstIterator it;
   if ( instrContext.findBurstByKey( key, it ) ) instrContext.removeBurst( it ); 
}

void Instrumentor::createStateEvent ( Event &e, nanos_event_state_value_t state )
{
   /* Registering a state event in instrucmentor context */
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.pushState(state);

   /* Creating a state event */
   e = State(state);
}

void Instrumentor::returnPreviousStateEvent ( Event &e )
{
   /* Recovering a state event in instrumentor context */
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.popState();
   nanos_event_state_value_t state = instrContext.topState(); 

   /* Creating a state event */
   e = State(state);
}

void Instrumentor::createPointEvent ( Event &e, unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values )
{
   /* Creating an Event::KV vector */
   Event::KVList kvlist = new Event::KV[nkvs];

   /* Initializing kvlist elements */
   for ( unsigned int i = 0; i < nkvs; i++ ) {
      kvlist[i] = Event::KV ( keys[i], values[i] );
   }

   /* Creating a point event */
   e = Point ( nkvs, kvlist );

}

void Instrumentor::createPtPStart ( Event &e, nanos_event_domain_t domain, nanos_event_id_t id,
                      unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values )
{
   /* Creating an Event::KV vector */
   Event::KVList kvlist = new Event::KV[nkvs];

   /* Initializing kvlist elements */
   for ( unsigned int i = 0; i < nkvs; i++ ) {
      kvlist[i] = Event::KV ( keys[i], values[i] );
   }

   /* Creating a PtP (start) event */
   e = PtP ( true, domain, id, nkvs, kvlist );

}
 
void Instrumentor::createPtPEnd ( Event &e, nanos_event_domain_t domain, nanos_event_id_t id,
                      unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values )
{
   /* Creating an Event::KV vector */
   Event::KVList kvlist = new Event::KV[nkvs];

   /* Initializing kvlist elements */
   for ( unsigned int i = 0; i < nkvs; i++ ) {
      kvlist[i] = Event::KV ( keys[i], values[i] );
   }

   /* Creating a PtP (end) event */
   e = PtP ( false, domain, id, nkvs, kvlist );

}

void Instrumentor::enterStartUp ( void )
{
   /* Registering a state (start-up) event in instrucmentor context */
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.pushState(STARTUP);

   /* Creating a state event (start-up), spawning event */
   Event e = State(STARTUP);
   addEventList ( 1u, &e );
}

void Instrumentor::leaveStartUp ( void )
{
   /* Recovering a state event from instrumentor context */
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.popState();
   nanos_event_state_value_t state = instrContext.topState();

   /* Returning to previous state, spawning event */
   Event e = State( state );
   addEventList ( 1u, &e );
}

void Instrumentor::enterShutDown ( void )
{
   /* Registering a state (shut-down) event in instrucmentor context */
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.pushState(SHUTDOWN);

   /* Creating a state event (shut-down), spawning event */
   Event e = State(SHUTDOWN);
   addEventList ( 1u, &e );
}

void Instrumentor::leaveShutDown ( void )
{
   /* Recovering a state event from instrumentor context */
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.popState();
   nanos_event_state_value_t state = instrContext.topState();

   /* Returning to previous state, spawning event */
   Event e = State( state );
   addEventList ( 1u, &e );
}

