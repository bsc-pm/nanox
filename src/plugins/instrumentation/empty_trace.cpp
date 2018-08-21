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

#include "plugin.hpp"
#include "system.hpp"
#include "instrumentation.hpp"

namespace nanos {

class InstrumentationEmptyTrace: public Instrumentation 
{
#ifndef NANOS_INSTRUMENTATION_ENABLED
   public:
      // constructor
      InstrumentationEmptyTrace( ) : Instrumentation( ) {}
      // destructor
      ~InstrumentationEmptyTrace() {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void ) {}
      void finalize( void ) {}
      void disable( void ) {}
      void enable( void ) {}
      void addResumeTask( WorkDescriptor &w ) {}
      void addSuspendTask( WorkDescriptor &w, bool last ) {}
      void addEventList ( unsigned int count, Event *events ) {}
      void threadStart( BaseThread &thread ) {}
      void threadFinish ( BaseThread &thread ) {}
#else
   public:
      // constructor
      InstrumentationEmptyTrace( ) : Instrumentation( *new InstrumentationContext() ) {}
      // destructor
      ~InstrumentationEmptyTrace () {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void ) {}
      void finalize( void ) {}
      void disable( void ) {}
      void enable( void ) {}
      void addResumeTask( WorkDescriptor &w ) {}
      void addSuspendTask( WorkDescriptor &w, bool last ) {}
      void addEventList ( unsigned int count, Event *events )
      {
// XXX: Used as template to help instrumentation plugin programmers to generate their code
#if 0
         InstrumentationDictionary *iD = sys.getInstrumentation()->getInstrumentationDictionary();
         nanos_event_key_t wd_ptr   = true ? iD->getEventKey("create-wd-ptr") : 0xFFFFFFFF;
         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            nanos_event_type_t type = e.getType();
            nanos_event_key_t key = e.getKey();
            if ( key == 0 ) continue;
            switch ( type ) {
               case NANOS_POINT:
                  if ( key == wd_ptr ) {
                     WD *wd = (WD *) e.getValue();
                     fprintf(stderr,"NANOS++: Creating task id = %ld\n", (long int) wd->getId() );
                  }
                  break;
               default:
                  break;
            }
         }
#endif
      }
      void threadStart( BaseThread &thread ) {}
      void threadFinish ( BaseThread &thread ) {}
#endif
};

namespace ext {

class InstrumentationEmptyTracePlugin : public Plugin {
   public:
      InstrumentationEmptyTracePlugin () : Plugin("Instrumentation which doesn't generate any trace.",1) {}
      ~InstrumentationEmptyTracePlugin () {}

      void config( Config &cfg ) {}

      void init ()
      {
         sys.setInstrumentation( new InstrumentationEmptyTrace() );	
      }
};

} // namespace ext

} // namespace nanos

DECLARE_PLUGIN("instrumentation-empty_trace",nanos::ext::InstrumentationEmptyTracePlugin);
