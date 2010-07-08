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
#include <stack>
#include <list>

#include "instrumentorcontext.hpp"
#include "instrumentor.hpp"
#include "debug.hpp"

using namespace nanos;

void InstrumentationContext::insertBurst ( const Event &e )
{
   bool found = false;
   BurstList::iterator it;
   nanos_event_key_t key = e.getKVs()[0].first;

   /* if found an event with the same key in the main list, send it to the backup list */
   for ( it = _burstList.begin() ; !found && (it != _burstList.end()) ; it++ ) {
      Event::ConstKVList kvlist = (*it).getKVs();
      if ( kvlist[0].first == key  )
      {
         _burstBackup.splice ( _burstBackup.begin(), _burstList, it );
         found = true;
      }
   }

   /* insert the event into the list */
   _burstList.push_front ( e );

}

void InstrumentationContext::removeBurst ( InstrumentationContext::BurstIterator it )
{
   bool found = false;
   nanos_event_key_t key = (*it).getKVs()[0].first;

   /* remove event from the list */
   _burstList.erase ( it );

   /* if found an event with the same key in the backup list, recover it to the main list */
   for ( it = _burstBackup.begin() ; !found && (it != _burstBackup.end()) ; it++ ) {
      Event::ConstKVList kvlist = (*it).getKVs();
      if ( kvlist[0].first == key  )
      {
         _burstList.splice ( _burstList.begin(), _burstBackup, it );
         found = true;
      }
   }
}

void InstrumentationContextStackedBursts::insertBurst ( const Event &e )
{
   /* insert the event into the list */
   _burstList.push_front ( e );
}

void InstrumentationContextStackedBursts::removeBurst ( BurstIterator it )
{
   /* remove event from the list */
   _burstList.erase ( it );
}

