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

#ifndef __NANOS_INSTRUMENTOR_CTX_H
#define __NANOS_INSTRUMENTOR_CTX_H
#include <stack>
#include <list>

#include "instrumentationcontext_decl.hpp"
#include "debug.hpp"

namespace nanos {

#ifdef NANOS_INSTRUMENTATION_ENABLED

inline bool InstrumentationContext::findBurstByKey ( InstrumentationContextData *icd, nanos_event_key_t key,
                                                     InstrumentationContextData::BurstIterator &ret )
{
   bool found = false;
   InstrumentationContextData::BurstIterator it;

   for ( it = icd->_burstList.begin() ; !found && (it != icd->_burstList.end()) ; it++ ) {
      nanos_event_key_t ckey = (*it).getKey();
      if ( ckey == key  ) { ret = it; found = true;}
  }

   return found;
}

inline InstrumentationContextData::ConstBurstIterator InstrumentationContext::beginBurst( InstrumentationContextData *icd ) const
{
   return icd->_burstList.begin();
}

inline InstrumentationContextData::ConstBurstIterator InstrumentationContext::endBurst( InstrumentationContextData *icd ) const
{
   return icd->_burstList.end();
}

inline void InstrumentationContext::insertDeferredEvent ( InstrumentationContextData *icd, const Event &e )
{
   /* insert the event into the list */
   {
      LockBlock( icd->_deferredEventsLock );
      icd->_deferredEvents.push_front ( e );
   }
}

inline void InstrumentationContext::clearDeferredEvents ( InstrumentationContextData *icd )
{
   /* remove all events from the list */
   icd->_deferredEvents.clear();
}

inline size_t InstrumentationContext::getNumDeferredEvents( InstrumentationContextData *icd ) const
{
   return icd->_deferredEvents.size();
}
inline InstrumentationContextData::EventIterator InstrumentationContext::beginDeferredEvents( InstrumentationContextData *icd ) const
{
   return icd->_deferredEvents.begin();
}
inline InstrumentationContextData::EventIterator InstrumentationContext::endDeferredEvents( InstrumentationContextData *icd ) const
{
   return icd->_deferredEvents.end();
}

inline void InstrumentationContext::disableStateEvents ( InstrumentationContextData *icd ) { icd->_stateEventEnabled = false; }
inline void InstrumentationContext::enableStateEvents ( InstrumentationContextData *icd ) { icd->_stateEventEnabled = true; }

#endif

} // namespace nanos

#endif
