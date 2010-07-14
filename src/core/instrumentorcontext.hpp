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
#ifndef __NANOS_INSTRUMENTOR_CTX_H
#define __NANOS_INSTRUMENTOR_CTX_H
#include <stack>
#include <list>

#include "instrumentorcontext_decl.hpp"
//#include "instrumentor_decl.hpp"
#include "debug.hpp"

using namespace nanos;

//#ifdef NANOS_INSTRUMENTATION_ENABLED

inline void InstrumentationContext::pushState ( InstrumentationContextData *icd, nanos_event_state_value_t state )
{
   icd->_stateStack.push( state );
}

inline void InstrumentationContext::popState ( InstrumentationContextData *icd )
{
   if ( !(icd->_stateStack.empty()) ) icd->_stateStack.pop();
}

inline nanos_event_state_value_t InstrumentationContext::topState ( InstrumentationContextData *icd )
{
   if ( !(icd->_stateStack.empty()) ) return icd->_stateStack.top();
   else return ERROR;
}

inline bool InstrumentationContext::findBurstByKey ( InstrumentationContextData *icd, nanos_event_key_t key,
                                                     InstrumentationContextData::BurstIterator &ret )
{
   bool found = false;
   InstrumentationContextData::BurstIterator it;

   for ( it = icd->_burstList.begin() ; !found && (it != icd->_burstList.end()) ; it++ ) {
      Instrumentation::Event::ConstKVList kvlist = (*it).getKVs();
      if ( kvlist[0].first == key  ) { ret = it; found = true;}
   }

   return found;
}

inline size_t InstrumentationContext::getNumBursts( InstrumentationContextData *icd ) const
{
   return icd->_burstList.size();
}

inline InstrumentationContextData::ConstBurstIterator InstrumentationContext::beginBurst( InstrumentationContextData *icd ) const
{
   return icd->_burstList.begin();
}

inline InstrumentationContextData::ConstBurstIterator InstrumentationContext::endBurst( InstrumentationContextData *icd ) const
{
   return icd->_burstList.end();
}

inline void InstrumentationContext::disableStateEvents ( InstrumentationContextData *icd )
{
   icd->_stateEventEnabled = false;
}

inline void InstrumentationContext::enableStateEvents ( InstrumentationContextData *icd )
{
   icd->_stateEventEnabled = true;
}

inline bool InstrumentationContext::isStateEventEnabled ( InstrumentationContextData *icd )
{
   return icd->_stateEventEnabled;
}

inline nanos_event_state_value_t InstrumentationContext::getValidState ( InstrumentationContextData *icd )
{
   return icd->_validState;
}

inline void InstrumentationContext::setValidState ( InstrumentationContextData *icd, nanos_event_state_value_t state )
{
   icd->_validState = state;
}
//#endif

#endif
