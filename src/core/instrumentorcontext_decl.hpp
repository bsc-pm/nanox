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
#ifndef __NANOS_INSTRUMENTOR_CTX_DECL_H
#define __NANOS_INSTRUMENTOR_CTX_DECL_H
#include <stack>
#include <list>

#include "instrumentor_decl.hpp"
#include "debug.hpp"

namespace nanos {

   class InstrumentationContextData {
      public:
         typedef Instrumentation::Event                  Event;                /**< Class defined in instrumentor_decl.hpp */
         typedef Instrumentation::Burst                  Burst;                /**< Class defined in instrumentor_decl.hpp */
         typedef std::stack<nanos_event_state_value_t>   StateStack;           /**< Stack of state's values */
         typedef std::list<Event>                        BurstList;            /**< List of Events (Bursts) */
         typedef BurstList::const_iterator               ConstBurstIterator;   /**< InstrumentationContext const BurstIterator */
         typedef BurstList::iterator                     BurstIterator;        /**< InstrumentationContext BurstIterator */
      public: //FIXME
         StateStack                 _stateStack;             /**< Stack of states */
         bool                       _stateEventEnabled;      /**< Set state level, zero by default */
         nanos_event_state_value_t  _validState;             /**< Last valid states */
         BurstList                  _burstList;              /**< List of current opened bursts */
         BurstList                  _burstBackup;            /**< Backup list (non-active) of opened bursts */
      public:
         /*! \brief InstrumentationContextData copy constructor
          */
         explicit InstrumentationContextData(const InstrumentationContextData &icd) : _stateStack(), _stateEventEnabled(icd._stateEventEnabled),
                  _validState(icd._validState), _burstList(), _burstBackup() {}
         /*! \brief InstrumentationContextData copy constructor
          */
         explicit InstrumentationContextData(const InstrumentationContextData *icd) : _stateStack(), _stateEventEnabled(icd->_stateEventEnabled),
                  _validState(icd->_validState), _burstList(), _burstBackup() {}
         /*! \brief InstrumentationContextData constructor
          */
         InstrumentationContextData() :_stateStack(), _stateEventEnabled(true), _validState(ERROR), _burstList(), _burstBackup() { }
         /*! \brief InstrumentationContextData destructor
          */
         ~InstrumentationContextData() {}
   };

   class InstrumentationContext {
         friend class Instrumentation;
      public:
         typedef Instrumentation::Event                  Event;                /**< Class defined in instrumentor_decl.hpp */
      public:
         /*! \brief InstrumentationContext constructor
          */
         InstrumentationContext () {}
         /*! \brief InstrumentationContext destructor
          */
         virtual ~InstrumentationContext() {}
      public: //FIXME /* Only friend classes (Instrumentation) can use InstrumentationContext */
         /*! \brief Adds a state value into the state stack 
          */
         void pushState ( InstrumentationContextData *icd, nanos_event_state_value_t state ); 
         /*! \brief Removes top state from the state stack
          */
         void popState ( InstrumentationContextData *icd ); 
         /*! \brief Gets current state from top of stack
          */
         nanos_event_state_value_t topState ( InstrumentationContextData *icd );
         /*! \brief Inserts a Burst into the burst list
          *
          *  This function inserts a burst event in the burst list. If an event with the same type of that
          *  event was already in the list it will be moved to an internal backup list in order to guarantee
          *  just one event per type in the list.
          */
         virtual void insertBurst ( InstrumentationContextData *icd, const Event &e );
         /*! \brief Removes a Burst from the burst list
          *
          *  This function removes a burst event from the burst list. If an event with the same type of that
          *  event was in the backup list it will be recovered to main list.
          */
         virtual void removeBurst ( InstrumentationContextData *icd, InstrumentationContextData::BurstIterator it ); 
         /*! \brief Look for a specific event given its key value
          */
         bool findBurstByKey ( InstrumentationContextData *icd, nanos_event_key_t key, InstrumentationContextData::BurstIterator &ret );
         /*! \brief Get the number of bursts in the main list
          */
         size_t getNumBursts( InstrumentationContextData *icd ) const ; 
         /*! \brief Gets the starting element in the burst list
          */
         InstrumentationContextData::ConstBurstIterator beginBurst( InstrumentationContextData *icd ) const ; 
         /*! \brief Gets the last element in the burst list
          */
         InstrumentationContextData::ConstBurstIterator endBurst( InstrumentationContextData *icd ) const ; 
         /*! \brief Enable state events
          */
         void enableStateEvents ( InstrumentationContextData *icd ) ;
         /*! \brief Disable state events
          */
         void disableStateEvents ( InstrumentationContextData *icd ) ;
         /*! \brief Get state events status
          */
         bool isStateEventEnabled ( InstrumentationContextData *icd ) ;
         /*! \brief Get valid state
          */
         nanos_event_state_value_t getValidState ( InstrumentationContextData *icd ) ;
         /*! \brief Save current state as valid state
          */
         void setValidState ( InstrumentationContextData *icd, nanos_event_state_value_t state ) ;
         /*!
          */
         virtual bool showStackedBursts( void );
         /*!
          */
         virtual bool showStackedState( void );
//#endif
   };

   class InstrumentationContextStackedBursts : public InstrumentationContext {
//#ifdef NANOS_INSTRUMENTATION_ENABLED
      public:
         InstrumentationContextStackedBursts () : InstrumentationContext() {}
         ~InstrumentationContextStackedBursts () {}
    
         bool showStackedBursts( void );
         bool showStackedState( void );
         void insertBurst ( InstrumentationContextData *icd, const Event &e );
         void removeBurst ( InstrumentationContextData *icd, InstrumentationContextData::BurstIterator it ); 
//#endif
   };
}
#endif
