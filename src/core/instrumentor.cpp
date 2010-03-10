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

#include "instrumentor_ctx.hpp"
#include "system.hpp"
#include "workdescriptor.hpp"

using namespace nanos;

#ifdef INSTRUMENTATION_ENABLED

/*! \brief Used by runtime API services to start a burst event and change state
 *
 *  \param [in] function is the funcition code used to create burst start
 *  \param [in] state is the state code we are starting now
 *
 *  \see Event Instrumentor::addEventList
 */
void Instrumentor::enterRuntimeAPI ( nanos_event_api_t function, nanos_event_state_value_t state )
{
   /* Create a vector of two events: STATE and BURST */
   Event::KV kv( Event::KV(NANOS_API,function) );
   Event e[2] = { State(state), Burst( true, kv) };

   /* Update instrumentor context with new state and open burst */
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.pushState(state);
   instrContext.insertBurst( e[1] );

   /* Spawning two events: specific instrumentor call */
   addEventList ( 2, e );
}

/*! \brief Used by runtime API services to close related burst event and coming back to previous state 
 *
 *  \see Event Instrumentor::addEventList
 */
void Instrumentor::leaveRuntimeAPI ( )
{
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   InstrumentorContext::BurstIterator it;
   if ( !instrContext.findBurstByKey( NANOS_API, it ) ) fatal0("Burst doesn't exists");

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

// FIXME (#140): Change InstrumentorContext ic.init() to Instrumentor::wdCreate();
void Instrumentor::wdCreate( WorkDescriptor* newWD )
{

}

/*! \brief Used by WorkDescriptor context switch. 
 *
 *  \param [in] oldWD is the WorkDescriptor leaving the cpu 
 *  \param [in] newWD is the WorkDescriptor entering the cpu
 *
 *  \see Event Instrumentor::addEventList
 */
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
   e[i++] = PtP (true,  NANOS_WD_DOMAIN, oldWD->getId(), 0, NULL);
   e[i++] = PtP (false, NANOS_WD_DOMAIN, newWD->getId(), 0, NULL);

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

/*! \brief Used by WorkDescriptor context switch when oldWD has finished execution
 *
 *  \param [in] oldWD is the WorkDescriptor leaving the cpu 
 *  \param [in] newWD is the WorkDescriptor entering the cpu
 *
 *  \see Event Instrumentor::addEventList
 */
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
   Event::KV kv( Event::KV( WD_ID, newWD->getId() ) );
   e[i++] = PtP ( false, 0, newWD->getId(), 1, &kv );

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

/*! \brief Used by idle WorkDescriptor in order to change thread state 
 *
 *  \see Event Instrumentor::addEventList
 */
void Instrumentor::enterIdle ( )
{
   InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
   instrContext.pushState(IDLE);

   Event e = State(IDLE);
   addEventList ( 1u, &e );
}

/*! \brief Used by idle WorkDescriptor in order to change thread state 
 *
 *  This function should be used only at the end of runtime execution
 *
 *  \see Event Instrumentor::addEventList
 */
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


#endif

