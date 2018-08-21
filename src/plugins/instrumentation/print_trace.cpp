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

#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "plugin.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"
#include "nanos-int.h"
#include "smpdd.hpp" // FIXME: this the include should not be here (just testing smpdd)

namespace nanos {

class InstrumentationPrintTrace: public Instrumentation 
{
#ifndef NANOS_INSTRUMENTATION_ENABLED
   public:
      // constructor
      InstrumentationPrintTrace() : Instrumentation() {}
      // destructor
      ~InstrumentationPrintTrace() {}

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
      InstrumentationPrintTrace() : Instrumentation( *new InstrumentationContextDisabled() ) {}
      // destructor
      ~InstrumentationPrintTrace ( ) {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void ) {}
      void finalize( void ) {}
      void disable( void ) {}
      void enable( void ) {}
      void addResumeTask( WorkDescriptor &w ) {
         fprintf(stderr,"NANOS++: (WD's) Resuming task %d in thread %d",w.getId(), myThread->getId());

         if (w.isImplicit() || w.getId() <= sys.getNumThreads() + 1) {
            fprintf( stderr,"\n" );
            return;
         }

         DOSubmit * doS = w.getDOSubmit();
         DependableObject::TargetVector::const_iterator it;

         if (doS != NULL) {
            const DependableObject::TargetVector &rt = doS->getReadTargets();
            fprintf( stderr,"; reads = " );
            for ( it = rt.begin(); it != rt.end(); ++it ) {
               fprintf(stderr,"%p, ",((*it)->getAddress()));
            }

            const DependableObject::TargetVector &wt = doS->getWrittenTargets();
            fprintf( stderr,"; writes = " );
            for (it = wt.begin(); it != wt.end(); ++it) {
               fprintf(stderr,"%p, ",((*it)->getAddress()));
            }
         }
         fprintf( stderr,"\n" );
      }

      void addSuspendTask( WorkDescriptor &w, bool last ) {
         fprintf(stderr,"NANOS++: (WD's) %s task %d in thread %d\n",last?"Finishing":"Suspending",w.getId(), myThread->getId());
      }
      
      void addEventList ( unsigned int count, Event *events )
      {
         // Getting instrumented key's
         InstrumentationDictionary *iD = sys.getInstrumentation()->getInstrumentationDictionary();
         char dirStr[12][20] = {"N/A" /*00*/, "RaW" /*01*/, "WaR" /*02*/, "WaW" /*03*/,
                                "CloseConcurrent" /*04*/, "OpenConcurrent" /*05*/,
                                "CloseCommutative" /*06*/, "OpenCommutative" /*07*/,
                                "CloseGeneral" /*08*/, "OpenGeneral" /*09*/, "Error" /*10*/, "" /*11*/};

         // Use following list of declaration to enable/disable event capture
         nanos_event_key_t create_task      = true ? iD->getEventKey("create-wd-ptr") : 0xFFFFFFFF;
         nanos_event_key_t funct_location   = true ? iD->getEventKey("user-funct-location") : 0xFFFFFFFF;
         nanos_event_key_t dependence       = true ? iD->getEventKey("dependence") : 0xFFFFFFFF;
         nanos_event_key_t dep_direction    = true ? iD->getEventKey("dep-direction") : 0xFFFFFFFF;
         nanos_event_key_t dep_address      = true ? iD->getEventKey("dep-address") : 0xFFFFFFFF;
         nanos_event_key_t nanos_api        = true ? iD->getEventKey("api") : 0xFFFFFFFF;
         nanos_event_key_t wd_id_event      = true ? iD->getEventKey("wd-id") : 0xFFFFFFFF;
         nanos_event_key_t user_code        = true ? iD->getEventKey("user-code") : 0xFFFFFFFF;
         nanos_event_key_t wd_ready         = true ? iD->getEventKey("wd-ready") : 0xFFFFFFFF;
         nanos_event_key_t wd_blocked       = true ? iD->getEventKey("wd-blocked") : 0xFFFFFFFF;

         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            nanos_event_type_t type = e.getType();
            nanos_event_key_t key = e.getKey();
            if ( key == 0 ) continue;
            int64_t value = e.getValue();
            switch ( type ) {
               case NANOS_POINT:
                  if ( key == create_task ) {
                     WorkDescriptor *wd = (WorkDescriptor *) value;
                     int64_t wd_id = wd->getId();
                     int64_t funct_id = (int64_t) wd->getActiveDevice().getWorkFct();
                     fprintf(stderr,"NANOS++: (WD) Executing %" PRId64 " function within task %" PRId64 " in thread %d\n",
                             funct_id, wd_id, myThread->getId());
                  }
                  if ( key == dependence ) {
                     nanos_event_value_t dependence_value = (events[i]).getValue();
                     int sender_id = (int) ( dependence_value >> 32 );
                     int receiver_id = (int) ( dependence_value & 0xFFFFFFFF );

                     // Getting dep_address event (usually the following event to dependence)
                     void * address_id = 0;
                     if ( dep_address != 0xFFFFFFFF ) {
                        unsigned int j = i;
                        while ( (j < count) && ((nanos_event_key_t)(events[j]).getKey() != dep_address) ) j++;
                        if ( j < count ) address_id = (void *) ((events[j]).getValue());
                     }
                     // Getting dep_address event (usually the following event to dependence)
                     int direction = 0;
                     if ( dep_direction != 0xFFFFFFFF ) {
                        unsigned int j = i;
                        while ( (j < count) && ((nanos_event_key_t)(events[j]).getKey() != dep_direction) ) j++;
                        if ( j < count ) direction = ( int ) ((events[j]).getValue());
                     }

                     if ( direction == 4 || direction == 6 || direction == 8 )
                        fprintf(stderr,"NANOS++: (DEP) Adding (%s) dependence %d->(%d) (related data address %p)\n",dirStr[direction],sender_id,receiver_id, address_id );
                     else if ( direction == 5 || direction == 7 || direction == 9 )
                        fprintf(stderr,"NANOS++: (DEP) Adding (%s) dependence (%d)->%d (related data address %p)\n",dirStr[direction],sender_id,receiver_id, address_id );
                     else
                        fprintf(stderr,"NANOS++: (DEP) Adding (%s) dependence %d->%d (related data address %p)\n",dirStr[direction],sender_id,receiver_id, address_id );

                  }
                  if ( key == wd_ready ) {
                     fprintf(stderr,"NANOS++: (WD-STATE) Task %d [%s] becomes ready\n", ((WD*)value)->getId(), ((WD*)value)->getDescription() );
                  }
                  if ( key == wd_blocked ) {
                     fprintf(stderr,"NANOS++: (WD-STATE) Task %d [%s] becomes blocked\n", ((WD*)value)->getId(), ((WD*)value)->getDescription()  );
                  }
                  break;
               case NANOS_BURST_START:
                  if ( key == wd_id_event ) {
                     fprintf(stderr,"NANOS++: (WD-ID) Entering %ld Work Descriptor\n", (long) value );
                  }
                  if ( key == user_code ) {
                     fprintf(stderr,"NANOS++: (USER-CODE) Entering %ld User Code Function\n", (long) value );
                  }
                  if ( key == nanos_api ) {
                     std::string description = iD->getValueDescription( e.getKey(), e.getValue() );
                     fprintf(stderr,"NANOS++: (API) Entering %ld named %s\n", (long) value, description.c_str() );
                  }
                  if ( key == funct_location ) {
                     std::string description = iD->getValueDescription( e.getKey(), e.getValue() );
                     fprintf(stderr,"NANOS++: (TASK) Executing %s function location\n", description.c_str() );
                  }
                  break;
               case NANOS_BURST_END:
                  if ( key == wd_id_event ) {
                     fprintf(stderr,"NANOS++: (WD-ID) Exiting %ld Work Descriptor\n", (long) value );
                  }
                  if ( key == user_code ) {
                     fprintf(stderr,"NANOS++: (USER-CODE) Exiting %ld User Code Function\n", (long) value );
                  }
                  if ( key == nanos_api ) {
                     std::string description = iD->getValueDescription( e.getKey(), e.getValue() );
                     fprintf(stderr,"NANOS++: (API) Exiting %ld named %s\n", (long) value, description.c_str() );
                  }
                  if ( key == funct_location ) {
                     std::string description = iD->getValueDescription( e.getKey(), e.getValue() );
                     fprintf(stderr,"NANOS++: (TASK) Finishing %s function location\n", description.c_str() );
                  }
                  break;
               case NANOS_PTP_START:
                  if ( key != 0xFFFFFFFF ) {
                  fprintf(stderr,"NANOS++: (P2P Start) Start p2p with key %u, %u, %llu\n", (unsigned int) e.getKey() , (unsigned int) e.getDomain(), (long long) e.getId() );                   
                  }
                  break;
               case NANOS_PTP_END:      
                  if ( key != 0xFFFFFFFF ) {
                     fprintf(stderr,"NANOS++: (P2P End) end p2p with key %u, %u, %llu\n", (unsigned int) e.getKey() , (unsigned int) e.getDomain(), (long long) e.getId() );                   
                  }
                  break;
               default:
                  break;
            }
         }
      }
      void threadStart( BaseThread &thread ) {}
      void threadFinish ( BaseThread &thread ) {}
#endif

};

namespace ext {

class InstrumentationPrintTracePlugin : public Plugin {
   public:
      InstrumentationPrintTracePlugin () : Plugin("Instrumentation which print the trace to std out.",1) {}
      ~InstrumentationPrintTracePlugin () {}

      void config( Config &cfg ) {}

      void init ()
      {
         sys.setInstrumentation( new InstrumentationPrintTrace() );	
      }
};

} // namespace ext

} // namespace nanos

DECLARE_PLUGIN("instrumentation-print_trace",nanos::ext::InstrumentationPrintTracePlugin);
