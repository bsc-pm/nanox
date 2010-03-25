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

#include "instrumentor.hpp"
#include "debug.hpp"

namespace nanos {

   class InstrumentorContext {
#ifdef INSTRUMENTATION_ENABLED
      private:
         typedef Instrumentor::Event Event;
         typedef Instrumentor::Burst Burst;
         typedef std::stack<nanos_event_state_value_t> StateStack;
         typedef std::list<Event> BurstList;

         StateStack       _stateStack;  /**< Stack of states */
         BurstList        _burstList;   /**< List of current opened bursts */
         BurstList        _burstBackup; /**< Backup list (non-active) of opened bursts */

      public:
         /*! \brief InstrumenotrContext const BurstIterator
          */
         typedef BurstList::const_iterator   ConstBurstIterator;

         /*! \brief InstrumenotrContext (non-const) BurstIterator
          */
         typedef BurstList::iterator         BurstIterator;

         /*! \brief InstrumentorContext copy constructor
          */
         explicit InstrumentorContext(const InstrumentorContext &ic) : _stateStack(), _burstList() { }

         /*! \brief InstrumentorContext constructor
          */
         InstrumentorContext () :_stateStack(), _burstList() { }

         /*! \brief InstrumentorContext destructor
          */
         ~InstrumentorContext() {}

         /*! \brief Initializes the InstrumentContext
          */
         void init ( unsigned int wd_id );

// xteruel:FIXME private and Instrumentor as friend class???

         /*! \brief Adds a state value into the state stack 
          */
         void pushState ( nanos_event_state_value_t state ); 

         /*! \brief Removes top state from the state stack
          */
         void popState ( void ); 

         /*! \brief Gets current state from top of stack
          */
         nanos_event_state_value_t topState ( void );

         /*! \brief Inserts a Burst into the burst list
          *
          *  This function inserts a burst event in the burst list. If an event with the same type of that
          *  event was already in the list it will be moved to an internal backup list in order to guarantee
          *  just one event per type in the list.
          */
         void insertBurst ( const Event &e );

         /*! \brief Removes a Burst from the burst list
          *
          *  This function removes a burst event from the burst list. If an event with the same type of that
          *  event was in the backup list it will be recovered to main list.
          */
         void removeBurst ( BurstIterator it ); 

         /*! \brief Look for a specific event given its key value
          */
         bool findBurstByKey ( nanos_event_key_t key, BurstIterator &ret );


         /*! \brief Get the number of bursts in the main list
          */
         unsigned int getNumBursts() const ; 

         /*! \brief Gets the starting element in the burst list
          */
         ConstBurstIterator beginBurst() const ; 

         /*! \brief Gets the last element in the burst list
          */
         ConstBurstIterator endBurst() const ; 

#else
         void init ( unsigned int wd_id ) {}
#endif

   };
}
#endif
