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

#include "instrumentor_decl.hpp"
#include "debug.hpp"

namespace nanos {
   class InstrumentorContext {
#ifdef INSTRUMENTATION_ENABLED
      private:
         typedef Instrumentor::Event Event;
         typedef Instrumentor::Burst Burst;
         typedef std::stack<nanos_event_state_t> StateStack;
         typedef std::list<Event> BurstList;

         StateStack       _stateStack;
         BurstList        _burstList;

      public:
         explicit InstrumentorContext(const InstrumentorContext &ic) : _stateStack(), _burstList() { }

         typedef BurstList::const_iterator BurstIterator;

         // constructors
         InstrumentorContext () :_stateStack(), _burstList() 
         {
         }
         ~InstrumentorContext() {}

         void pushState ( nanos_event_state_t state ) { _stateStack.push( state ); }
         void popState ( void ) { if ( !(_stateStack.empty()) ) _stateStack.pop(); }

         nanos_event_state_t topState ( void )
         {
            if ( !(_stateStack.empty()) ) return _stateStack.top();
            else return ERROR;
         }

         void pushBurst ( const Event &e )
         {
            _burstList.push_front ( e );
         }

         void popBurst ( void )
         {
            if ( !(_burstList.empty()) ) _burstList.pop_front ( );
            else fatal0("Instrumentor burst error (empty burst list).");
            // FIXME: previous line should be fatal. Fix when we can include system.hpp
            // else fatal("Instrumentor burst error (empty burst list).");
         }

         Event & topBurst ( void )
         {
            if ( !(_burstList.empty()) ) return _burstList.front();
            fatal0("Instrumentor burst error (empty burst list).");
            // FIXME: previous line should be fatal. Fix when we can include system.hpp
         }
          
         bool findBurstByKey ( Event::Key key )
         {
            bool find = false;

            for ( BurstList::iterator it = _burstList.begin() ; !find && (it != _burstList.end()) ; it++ ) {
               Event::ConstKVList kvlist = (*it).getKVs();
               if ( kvlist[0].first == key  )
               {
                  find = true;
                  if ( it != _burstList.begin() ) _burstList.splice ( _burstList.begin(), _burstList, it );
               }
            }
            return find;
         }

         unsigned int getNumBursts() const { return _burstList.size(); }
         BurstIterator beginBurst() const { return _burstList.begin(); }
         BurstIterator endBurst() const { return _burstList.end(); }

         void init ( unsigned int wd_id )
         {
            Event::KV kv( Event::KV( Event::WD_ID, wd_id ) );
            Event e = Burst( kv );
 
            pushBurst( e );
            pushState( RUNNING );
         }
#else
      public:
         void init ( unsigned int wd_id ) { }
#endif
   };
}
#endif
