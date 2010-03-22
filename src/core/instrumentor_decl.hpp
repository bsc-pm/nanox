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

#ifndef __NANOS_INSTRUMENTOR_DECL_H
#define __NANOS_INSTRUMENTOR_DECL_H
#include <list>
#include <utility>
#include "debug.hpp"
#include "nanos-int.h"
#include "workdescriptor_fwd.hpp"

namespace nanos {

   class Instrumentor 
   {
      public:
         class Event {
            public:
               typedef std::pair<nanos_event_key_t,nanos_event_value_t>   KV;
               typedef KV *KVList;
               typedef const KV *ConstKVList;
            protected:
               nanos_event_type_t          _type;         /**< Event type */
               nanos_event_state_value_t   _state;        /**< Event state */

               unsigned int                _nkvs;         /**< Number of kvs elements */
               KVList                      _kvList;       /**< List of elements kv (key.value) */
               bool                        _kvListOwner;  /**< Is the object the owner of kvList */

               nanos_event_domain_t        _ptpDomain;    /**< A specific domain in which ptpId is unique */
               nanos_event_id_t            _ptpId;        /**< PtP event id */

            public:
               /*! \brief Event constructor (generic constructor used by all other specific constructors)
                *  \see State Burst Point PtP
                */
               Event ( nanos_event_type_t type, nanos_event_state_value_t state, unsigned int nkvs, KVList kvlist, bool kvlist_owner,
                       nanos_event_domain_t ptp_domain, nanos_event_id_t ptp_id ) :
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

               /*! \brief Event copy constructor
                */
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

               /*! \brief Event assignment operator
                */
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

               /*! \brief Event destructor
                */
               ~Event() { if ( _kvListOwner ) delete[] _kvList; }

               /*! \brief Get event type
                */
               nanos_event_type_t getType () const; 

               /*! \brief Get event state
                */
               nanos_event_state_value_t getState ();

               /*! \brief Get number of kvs
                */
               unsigned int getNumKVs () const;

               /*! \brief Get kvs vector
                */
               ConstKVList getKVs () const;

               /*! \brief Get specific domain ( useful in PtP events)
                *  \see getId
                */
               unsigned int getDomain ( void ) const;

               /*! \brief Get event id (unique in a specific domain, useful in PtP events)
                *  \see getDomain
                */
               unsigned int getId( void ) const;

               /*! \brief Change event type to the complementary value (i.e. if type is BURST_START it changes to BURST_END)
                */
               void reverseType ( );
         };

         class State : public Event 
         {
            public:
              /*! \brief State event constructor
               */
              State ( nanos_event_state_value_t state = ERROR ) 
                 : Event (STATE, state, 0, NULL, false, (nanos_event_domain_t) 0, (nanos_event_id_t) 0 ) { }
         };

         class Burst : public Event 
         {
             public:
               /*! \brief Burst event constructor
                */
               Burst ( bool start, KV kv )
                 : Event ( start? BURST_START: BURST_END, ERROR, 1, &kv, false, (nanos_event_domain_t) 0, (nanos_event_id_t) 0 ) { }

         };

         class Point : public Event 
         {
             public:
               /*! \brief Point event constructor
                */
               Point ( unsigned int nkvs, KVList kvlist )
                 : Event ( POINT, ERROR, nkvs, kvlist, false, (nanos_event_domain_t) 0, (nanos_event_id_t) 0 ) { }
         };

         class PtP : public Event 
         {
            public:
               /*! \brief PtP event constructor
                */
               PtP ( bool start, nanos_event_domain_t domain, nanos_event_id_t id, unsigned int nkvs,  KVList kvlist )
                  : Event ( start ? PTP_START : PTP_END , ERROR, nkvs, kvlist, false, domain, id ) { }

         };

      public:
         /*! \brief Instrumentor constructor
          */
         Instrumentor() {}

         /*! \brief Instrumentor destructor
          */
         virtual ~Instrumentor() {}

#ifdef INSTRUMENTATION_ENABLED

       // low-level instrumentation interface (pure virtual functions)

       /*! \brief Pure virtual functions executed at the beginning of instrumentation phase
        *
        *  Each of (specific) instrumentor modules have to implement this function in order
        *  to be consistent with the instrumentation model
        */
       virtual void initialize( void ) = 0;

       /*! \brief Pure virtual functions executed at the end of instrumentation phase
        *
        *  Each of (specific) instrumentor modules have to implement this function in order
        *  to be consistent with the instrumentation model
        */
       virtual void finalize( void ) = 0;

       /*! \brief Pure virtual functions executed each time runtime wants to add an event
        *
        *  Each of (specific) instrumentor modules have to implement this function in order
        *  to be consistent with the instrumentation model. This function includes several
        *  events in a row to facilitate implementation in which several events occurs at
        *  the same time (i.e. same timestamp).
        */
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
       void createStateEvent ( Event &e, nanos_event_state_value_t state );
       void returnPreviousStateEvent ( Event &e );
       void createPointEvent ( Event &e, unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values );
       void createPtPStart ( Event &e, nanos_event_domain_t domain, nanos_event_id_t id,
                             unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values );
       void createPtPEnd ( Event &e, nanos_event_domain_t domain, nanos_event_id_t id,
                             unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values );

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
       void createStateEvent ( Event &e, nanos_event_state_value_t state ) {}
       void returnPreviousStateEvent ( Event &e ) {}
       void createPointEvent ( Event &e, unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values ) {}
       void createPtPStart ( Event &e, nanos_event_domain_t domain, nanos_event_id_t id,
                             unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values ) {}
       void createPtPEnd ( Event &e, nanos_event_domain_t domain, nanos_event_id_t id,
                             unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values ) {}

#endif

  };


}
#endif
