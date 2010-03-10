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
//#define INSTRUMENTATION_ENABLED

#ifndef __NANOS_INSTRUMENTOR_DECL_H
#define __NANOS_INSTRUMENTOR_DECL_H
#include <list>
#include <utility>
#include "debug.hpp"
#include "nanos-int.h"

namespace nanos {

// forward decl
   class WorkDescriptor;

   class Instrumentor {
      public:
         class Event {
            public:
               typedef std::pair<nanos_event_key_t,nanos_event_value_t>   KV;
               typedef KV*                                                KVList;
               typedef const KV *                                         ConstKVList;
            protected:
               nanos_event_type_t          _type;
               nanos_event_state_value_t   _state;

               unsigned int          _nkvs;
               KVList                _kvList;
               bool                  _kvListOwner;

               unsigned int          _ptpDomain;
               unsigned int          _ptpId;

            public:
               Event ( nanos_event_type_t type, nanos_event_state_value_t state, unsigned int nkvs, KVList kvlist, bool kvlist_owner,
                       unsigned int ptp_domain, unsigned int ptp_id ) :
                     _type (type), _state (state), _nkvs(nkvs), _kvList (kvlist), _kvListOwner(kvlist_owner),
                     _ptpDomain (ptp_domain), _ptpId (ptp_id)
               {
                  if ( _type == BURST_START || _type == BURST_END )
                  {
                     _kvList = new KV[1];
                     _kvList[0] = *kvlist;
                     _kvListOwner = true;
                  }
               }

               Event ( const Event & evt )
               {
                  _type = evt._type;
                  _state = evt._state;
                  _nkvs = evt._nkvs;
                  _kvList = new KV[_nkvs];
                  for ( unsigned int i = 0; i < _nkvs; i++ ) {
                     _kvList[i] = evt._kvList[i];
                  }
                  _kvListOwner = true;
                  _ptpDomain = evt._ptpDomain;
                  _ptpId     = evt._ptpId;

               }

               void operator= ( const Event & evt ) 
               { 
                  // self-assignment: ok
                  if ( this == &evt ) return; 
                    
                  _type = evt._type;
                  _state = evt._state;
                  _nkvs = evt._nkvs;
                  _kvList = new KV[_nkvs];
                  for ( unsigned int i = 0; i < _nkvs; i++ ) {
                     _kvList[i] = evt._kvList[i];
                  }
                  _kvListOwner = true;
                  _ptpDomain = evt._ptpDomain;
                  _ptpId     = evt._ptpId;

               }

               ~Event() { if ( _kvListOwner ) delete[] _kvList; }

               
               nanos_event_type_t getType () const { return _type; } 

               nanos_event_state_value_t getState () { return _state; }

               unsigned int getNumKVs () const { return _nkvs; }
               ConstKVList getKVs () const { return _kvList; } 

               unsigned int getDomain ( void ) const { return _ptpDomain; }
               unsigned int getId( void ) const { return _ptpId; }

               void reverseType ( )
               {
                   switch ( _type )
                   {
                      case PTP_START: _type = PTP_END; break;
                      case PTP_END: _type = PTP_START; break;
                      case BURST_START: _type = BURST_END; break;
                      case BURST_END: _type = BURST_START; break;
                      default: break;
                   }
               }
         };

         class State : public Event {
            public:
              State ( nanos_event_state_value_t state = ERROR ) 
                 : Event (STATE, state, 0, NULL, false, 0, 0 ) { }
         };

         class Burst : public Event {
             public:
               Burst ( bool start, KV kv )
                 : Event ( start? BURST_START: BURST_END, ERROR, 1, &kv, false, 0, 0 ) { }

         };

         class Point : public Event {
             public:
               Point ( unsigned int nkvs, KVList kvlist )
                 : Event ( POINT, ERROR, nkvs, kvlist, false, 0, 0 ) { }
         };

         class PtP : public Event {
            public:
               PtP ( bool start, unsigned int domain, unsigned int id, unsigned int nkvs,  KVList kvlist )
                  : Event ( start ? PTP_START : PTP_END , ERROR, nkvs, kvlist, false, domain, id ) { }

         };

      public:
         Instrumentor() {}
         virtual ~Instrumentor() {}

#ifdef INSTRUMENTATION_ENABLED

       // low-level instrumentation interface (pure virtual functions)

       virtual void initialize( void ) = 0;
       virtual void finalize( void ) = 0;
       virtual void addEventList ( unsigned int count, Event *events ) = 0;

       // CORE: high-level instrumentation interface (virtual functions)

       virtual void enterRuntimeAPI ( nanos_event_api_t function, nanos_event_state_value_t state = RUNTIME );
       virtual void leaveRuntimeAPI ( );
       virtual void enterIdle ( );
       virtual void leaveIdle ( );

       virtual void wdCreate( WorkDescriptor* newWD );
       virtual void wdSwitch( WorkDescriptor* oldWD, WorkDescriptor* newWD );
       virtual void wdExit( WorkDescriptor* oldWD, WorkDescriptor* newWD );

       virtual void enterStartUp ( void );
       virtual void leaveStartUp ( void );
       virtual void enterShutDown ( void );
       virtual void leaveShutDown ( void );

       // CORE: high-level instrumentation interface (non-virtual functions)
       void createBurstStart ( Event &e, nanos_event_key_t key, nanos_event_value_t value );
       void createBurstEnd ( Event &e, nanos_event_key_t key, nanos_event_value_t value );

#else

       // All functions here must be empty and  non-virtual so the compiler 
       // eliminates the instrumentation calls

       // low-level instrumentation interface (pure virtual functions)

       void initialize( void ) {} 
       void finalize( void ) {}
       void addEventList ( unsigned int count, Event *events ) {}

       // CORE: high-level instrumentation interface (virtual functions)

       void enterRuntimeAPI ( nanos_event_api_t function, nanos_event_state_value_t state = RUNTIME ) {}
       void leaveRuntimeAPI ( ) {}
       void enterIdle ( ) {}
       void leaveIdle ( ) {}

       void wdCreate( WorkDescriptor* newWD ) {}
       void wdSwitch( WorkDescriptor* oldWD, WorkDescriptor* newWD ) {}
       void wdExit( WorkDescriptor* oldWD, WorkDescriptor* newWD ) {}

       void enterStartUp ( void ) {}
       void leaveStartUp ( void ) {}
       void enterShutDown ( void ) {}
       void leaveShutDown ( void ) {}


       // CORE: high-level instrumentation interface (non-virtual functions)
       void createBurstStart ( Event &e, nanos_event_key_t key, nanos_event_value_t value ) {}
       void createBurstEnd ( Event &e, nanos_event_key_t key, nanos_event_value_t value ) {}

#endif

  };


}
#endif
