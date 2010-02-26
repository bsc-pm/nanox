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
// FIXME: (#131) This flag ENABLE_INSTRUMENTATION has to be managed through
//compilation in order to generate an instrumentation version
#define INSTRUMENTATION_ENABLED

#ifndef __NANOS_INSTRUMENTOR_H
#define __NANOS_INSTRUMENTOR_H
#include "instrumentor_decl.hpp"
#include "instrumentor_ctx.hpp"
#include "workdescriptor.hpp"

namespace nanos {

#ifdef INSTRUMENTATION_ENABLED

       // CORE: high-level instrumentation interface (virtual functions)
       void Instrumentor::enterRuntimeAPI ( nanos_event_api_t function, nanos_event_state_t state )
       {
          Event::KV kv( Event::KV(Event::NANOS_API,function) );
          Event e[2] = { State(state), Burst( kv) };
          InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
          instrContext.pushState(state);
          instrContext.pushBurst(e[1]);
          addEventList ( 2, e );
       }

       void Instrumentor::leaveRuntimeAPI ( )
       {
          InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
          Event &e1 = instrContext.topBurst();
          e1.reverseType();
          /* Top is current state, so before we have to pop previous state
           * on top of the stack and then restore previous state */
          instrContext.popState();
          nanos_event_state_t state = instrContext.topState();
          Event e[2] = { State(state), e1 };
          addEventList ( 2, e );

          instrContext.popBurst(); 
       }

       // FIXME (140): Change InstrumentorContext ic.init() to Instrumentor::wdCreate();
       void Instrumentor::wdCreate( WorkDescriptor* newWD ) {}

       void Instrumentor::wdSwitch( WorkDescriptor* oldWD, WorkDescriptor* newWD )
       {
          /* Computing number of burst events */
          InstrumentorContext &oldInstrContext = oldWD->getInstrumentorContext();
          InstrumentorContext &newInstrContext = newWD->getInstrumentorContext();
          unsigned int oldBursts = oldInstrContext.getNumBursts();
          unsigned int newBursts = newInstrContext.getNumBursts();
          unsigned int numEvents = 3 + oldBursts + newBursts;

          /* Allocating Events */
          Event *e = (Event *) alloca(sizeof(Event) * numEvents );

          /* Creating PtP events */
          e[0] = PtP (true,  PtP::WD, oldWD->getId(), 0, NULL);
          e[1] = PtP (false, PtP::WD, newWD->getId(), 0, NULL);

          /* Change state for newWD */
          nanos_event_state_t state = newInstrContext.topState();
          e[2] = State ( state );

          /* Regenerating reverse bursts for old WD */
          int i = 2;
          for ( InstrumentorContext::BurstIterator it = oldInstrContext.beginBurst() ;
                it != oldInstrContext.endBurst(); it++,i++ ) {
             e[i] = *it;
             e[i].reverseType();
          }

          /* Regenerating bursts for new WD */
          for ( InstrumentorContext::BurstIterator it = newInstrContext.beginBurst() ;
                it != newInstrContext.endBurst(); it++,i++ ) {
             e[i] = *it;
          }
          /* Adding event list */
          addEventList ( numEvents, e );
       }

       void Instrumentor::wdExit( WorkDescriptor* oldWD, WorkDescriptor* newWD )
       {

          /* Computing number of events */
          InstrumentorContext &oldInstrContext = oldWD->getInstrumentorContext();
          InstrumentorContext &newInstrContext = newWD->getInstrumentorContext();
          unsigned int oldBursts = oldInstrContext.getNumBursts();
          unsigned int newBursts = newInstrContext.getNumBursts();
          unsigned int numEvents = 2 + oldBursts + newBursts;

          /* Allocating Events */
          Event *e = (Event *) alloca(sizeof(Event) * numEvents );

          /* Creating PtP events */
          Event::KV kv( Event::KV( Event::WD_ID, newWD->getId() ) );
          e[0] = PtP ( false, 0, newWD->getId(), 1, &kv );

          /* Change state for newWD */
          nanos_event_state_t state = newInstrContext.topState();
          e[1] = State ( state );

          int i = 2; 
          /* Regenerating reverse bursts for old WD */
          for ( InstrumentorContext::BurstIterator it = oldInstrContext.beginBurst() ;
                it != oldInstrContext.endBurst(); it++,i++ ) {
             e[i] = *it;
             e[i].reverseType();
          }

          /* Regenerating bursts for new WD */
          for ( InstrumentorContext::BurstIterator it = newInstrContext.beginBurst() ;
                it != newInstrContext.endBurst(); it++,i++ ) {
             e[i] = *it;
          }

          /* Adding event list */
          addEventList ( numEvents, e );
       }

       void Instrumentor::enterIdle ( )
       {
          InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
          instrContext.pushState(IDLE);

          Event e = State(IDLE);
          addEventList ( 1u, &e );
       }

       /* This function should be used only at the end of runtime execution */
       void Instrumentor::leaveIdle ( )
       {
          InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
          instrContext.popState();
          nanos_event_state_t state = instrContext.topState();
          Event e = State( state );
          addEventList ( 1u, &e );
       }

#endif

}
#endif
