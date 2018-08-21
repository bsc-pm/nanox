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

#include <stack>
#include <list>

#include "instrumentationcontext.hpp"
#include "instrumentation.hpp"
#include "debug.hpp"

using namespace nanos;

#ifdef NANOS_INSTRUMENTATION_ENABLED


/* InstrumentationContext is default implementation: isContextSwitchEnabled */
bool InstrumentationContext::isContextSwitchEnabled ( void ) const { return true; }
bool InstrumentationContextDisabled::isContextSwitchEnabled ( void ) const { return false; }


/* InstrumentationContext is default implementation: showStackedBursts */
bool InstrumentationContext::showStackedBursts ( void ) { return false; }
bool InstrumentationContextStackedStates::showStackedBursts ( void ) { return false; }
bool InstrumentationContextStackedBursts::showStackedBursts ( void ) { return true; }
bool InstrumentationContextStackedStatesAndBursts::showStackedBursts ( void ) { return true; }
bool InstrumentationContextDisabled::showStackedBursts ( void ) { return false; }


/* InstrumentationContext is default implementation: showStackeStates */
bool InstrumentationContext::showStackedStates ( void ) { return false; }
bool InstrumentationContextStackedStates::showStackedStates ( void ) { return true; }
bool InstrumentationContextStackedBursts::showStackedStates ( void ) { return false; }
bool InstrumentationContextStackedStatesAndBursts::showStackedStates ( void ) { return true; }
bool InstrumentationContextDisabled::showStackedStates ( void ) { return false; }


/* InstrumentationContext is default implementation: insertBurst */
void InstrumentationContext::insertBurst ( InstrumentationContextData *icd, const Event &e )
{
   bool found = false;
   InstrumentationContextData::EventList::iterator it;
   nanos_event_key_t key = e.getKey();

   /* if found an event with the same key in the main list, send it to the backup list */
   for ( it = icd->_burstList.begin() ; !found && (it != icd->_burstList.end()) ; it++ ) {
      nanos_event_key_t ckey = (*it).getKey();
      if ( ckey == key  )
      {
         icd->_burstBackup.splice ( icd->_burstBackup.begin(), icd->_burstList, it );
         found = true;
      }
   }

   /* insert the event into the list */
   icd->_burstList.push_front ( e );

}
void InstrumentationContextStackedBursts::insertBurst ( InstrumentationContextData *icd, const Event &e )
{
   /* insert the event into the list */
   icd->_burstList.push_front ( e );
}
void InstrumentationContextStackedStatesAndBursts::insertBurst ( InstrumentationContextData *icd, const Event &e )
{
   /* insert the event into the list */
   icd->_burstList.push_front ( e );
}
void InstrumentationContextDisabled::insertBurst ( InstrumentationContextData *icd, const Event &e ) { }


/* InstrumentationContext is default implementation: removeBurst */
void InstrumentationContext::removeBurst ( InstrumentationContextData *icd, InstrumentationContextData::BurstIterator it )
{
   bool found = false;
   nanos_event_key_t key = (*it).getKey();

   /* remove event from the list */
   icd->_burstList.erase ( it );

   /* if found an event with the same key in the backup list, recover it to the main list */
   for ( it = icd->_burstBackup.begin() ; !found && (it != icd->_burstBackup.end()) ; it++ ) {
      nanos_event_key_t ckey = (*it).getKey();
      if ( ckey == key  )
      {
         icd->_burstList.splice ( icd->_burstList.begin(), icd->_burstBackup, it );
         found = true;
      }
   }
}
void InstrumentationContextStackedBursts::removeBurst ( InstrumentationContextData *icd, InstrumentationContextData::BurstIterator it )
{
   /* remove event from the list */
   icd->_burstList.erase ( it );
}
void InstrumentationContextStackedStatesAndBursts::removeBurst ( InstrumentationContextData *icd, InstrumentationContextData::BurstIterator it )
{
   /* remove event from the list */
   icd->_burstList.erase ( it );
}
void InstrumentationContextDisabled::removeBurst ( InstrumentationContextData *icd, InstrumentationContextData::BurstIterator it ) {}


/* InstrumentationContext is default implementation: pushState */
void InstrumentationContext::pushState ( InstrumentationContextData *icd, nanos_event_state_value_t state )
{
   icd->_stateStack.push_back( state );
}
void InstrumentationContextDisabled::pushState ( InstrumentationContextData *icd, nanos_event_state_value_t state ) {}


void InstrumentationContext::popState ( InstrumentationContextData *icd )
{
   if ( !(icd->_stateStack.empty()) ) icd->_stateStack.pop_back();
}
void InstrumentationContextDisabled::popState ( InstrumentationContextData *icd ) {}


nanos_event_state_value_t InstrumentationContext::topState ( InstrumentationContextData *icd )
{
   if ( !(icd->_stateStack.empty()) ) return icd->_stateStack.back();
   else return NANOS_ERROR;
}
nanos_event_state_value_t InstrumentationContextDisabled::topState ( InstrumentationContextData *icd ) { return NANOS_ERROR; }

nanos_event_state_value_t InstrumentationContext::getState ( InstrumentationContextData *icd )
{
   if ( !(icd->_stateStack.empty()) ) return icd->_stateStack.back();
   else return NANOS_ERROR;
}
nanos_event_state_value_t InstrumentationContextDisabled::getState ( InstrumentationContextData *icd ) { return NANOS_ERROR; }

size_t InstrumentationContext::getStateStackSize ( InstrumentationContextData *icd ) { return (size_t) icd->_stateStack.size(); }
size_t InstrumentationContextDisabled::getStateStackSize ( InstrumentationContextData *icd ) { return (size_t) 0; }

size_t InstrumentationContext::getNumBursts( InstrumentationContextData *icd ) const { return icd->_burstList.size(); }
size_t InstrumentationContextDisabled::getNumBursts( InstrumentationContextData *icd ) const { return (size_t) 0; }

size_t InstrumentationContext::getNumStates( InstrumentationContextData *icd ) const { return icd->_stateStack.size() < 1 ? 0 : 1; }
size_t InstrumentationContextStackedStates::getNumStates( InstrumentationContextData *icd ) const { return icd->_stateStack.size(); }
size_t InstrumentationContextStackedStatesAndBursts::getNumStates( InstrumentationContextData *icd ) const { return icd->_stateStack.size(); }
size_t InstrumentationContextDisabled::getNumStates( InstrumentationContextData *icd ) const { return (size_t) 0; }

InstrumentationContextData::ConstStateIterator InstrumentationContext::beginState( InstrumentationContextData *icd ) const
{
   if ( !(icd->_stateStack.empty()) ) return --(icd->_stateStack.end());
   else return icd->_stateStack.end();
}
InstrumentationContextData::ConstStateIterator InstrumentationContextStackedStates::beginState( InstrumentationContextData *icd ) const
{
   return icd->_stateStack.begin();
}
InstrumentationContextData::ConstStateIterator InstrumentationContextStackedStatesAndBursts::beginState( InstrumentationContextData *icd ) const
{
   return icd->_stateStack.begin();
}
InstrumentationContextData::ConstStateIterator InstrumentationContextDisabled::beginState( InstrumentationContextData *icd ) const
{
   return icd->_stateStack.end();
}
InstrumentationContextData::ConstStateIterator InstrumentationContext::endState( InstrumentationContextData *icd ) const
{
   return icd->_stateStack.end();
}
#endif
