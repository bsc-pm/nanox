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
         fprintf(stderr,"NANOS++: (WD's) Resuming task %d in thread %d\n",w.getId(), myThread->getId());
      }
      void addSuspendTask( WorkDescriptor &w, bool last ) {
         fprintf(stderr,"NANOS++: (WD's) %s task %d in thread %d\n",last?"Finishing":"Suspending",w.getId(), myThread->getId());
      }

      void addEventList ( unsigned int count, Event *events )
      {
         // Getting instrumented key's
         InstrumentationDictionary *iD = sys.getInstrumentation()->getInstrumentationDictionary();

         // Use following list of declaration to enable/disable event capture
         nanos_event_key_t create_task      = false ? iD->getEventKey("create-wd-ptr") : 0xFFFFFFFF;
         nanos_event_key_t funct_location   = false ? iD->getEventKey("user-funct-location") : 0xFFFFFFFF;
         nanos_event_key_t dependence       = false ? iD->getEventKey("dependence") : 0xFFFFFFFF;
         nanos_event_key_t dep_address      = false ? iD->getEventKey("dep-address") : 0xFFFFFFFF;
         nanos_event_key_t nanos_api        = true ? iD->getEventKey("api") : 0xFFFFFFFF;
         nanos_event_key_t wd_id_event      = true ? iD->getEventKey("wd-id") : 0xFFFFFFFF;
         nanos_event_key_t user_code        = true ? iD->getEventKey("user-code") : 0xFFFFFFFF;
         nanos_event_key_t wd_ready         = true ? iD->getEventKey("wd-ready") : 0xFFFFFFFF;
         nanos_event_key_t wd_blocked       = true ? iD->getEventKey("wd-blocked") : 0xFFFFFFFF;

         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            int64_t value = e.getValue();
            switch ( e.getType() ) {
               case NANOS_POINT:
                  if ( e.getKey() == create_task ) {
                     WorkDescriptor *wd = (WorkDescriptor *) value;
                     int64_t wd_id = wd->getId();
                     int64_t funct_id = (int64_t) ((ext::SMPDD &) (wd->getActiveDevice ())).getWorkFct ();
                     fprintf(stderr,"NANOS++: (WD) Executing %" PRId64 " function within task %" PRId64 " in thread %d\n",
                             funct_id, wd_id, myThread->getId());
                  }
                  if ( (nanos_event_key_t)(events[i]).getKey() == dependence ) {
                     nanos_event_value_t dependence_value = (events[i]).getValue();
                     int sender_id = (int) ( dependence_value >> 32 );
                     int receiver_id = (int) ( dependence_value & 0xFFFFFFFF );

                     // Getting dep_address event (usually the following event to dependence)
                     void * address_id = 0;
                     if ( dep_address != 0xFFFFFFFF ) {
                        while ( (i < count) && ((nanos_event_key_t)(events[i]).getKey() != dep_address) ) i++;
                        if ( i < count ) address_id = (void *) ((events[i]).getValue());
                     }

                     fprintf(stderr,"NANOS++: (DEP) Adding dependence %d->%d (related data address %p)\n",sender_id,receiver_id, address_id );
                  }
                  if ( e.getKey() == wd_ready ) {
                     fprintf(stderr,"NANOS++: (WD-STATE) Task %d becomes ready\n", ((WD *) value)->getId() );
                  }
                  if ( e.getKey() == wd_blocked ) {
                     fprintf(stderr,"NANOS++: (WD-STATE) Task %d becomes blocked\n", ((WD *) value)->getId() );
                  }
                  break;
               case NANOS_BURST_START:
                  if ( e.getKey() == wd_id_event ) {
                     fprintf(stderr,"NANOS++: (WD-ID) Entering %ld Work Descriptor\n", (long) value );
                  }
                  if ( e.getKey() == user_code ) {
                     fprintf(stderr,"NANOS++: (USER-CODE) Entering %ld User Code Function\n", (long) value );
                  }
                  if ( e.getKey() == nanos_api ) {
                     std::string description = iD->getValueDescription( e.getKey(), e.getValue() );
                     fprintf(stderr,"NANOS++: (API) Entering %ld named %s\n", (long) value, description.c_str() );
                  }
                  if ( e.getKey() == funct_location ) {
                     std::string description = iD->getValueDescription( e.getKey(), e.getValue() );
                     fprintf(stderr,"NANOS++: (TASK) Executing %s function location\n", description.c_str() );
                  }
                  break;
               case NANOS_BURST_END:
                  if ( e.getKey() == wd_id_event ) {
                     fprintf(stderr,"NANOS++: (WD-ID) Exiting %ld Work Descriptor\n", (long) value );
                  }
                  if ( e.getKey() == user_code ) {
                     fprintf(stderr,"NANOS++: (USER-CODE) Exiting %ld User Code Function\n", (long) value );
                  }
                  if ( e.getKey() == nanos_api ) {
                     std::string description = iD->getValueDescription( e.getKey(), e.getValue() );
                     fprintf(stderr,"NANOS++: (API) Exiting %ld named %s\n", (long) value, description.c_str() );
                  }
                  if ( e.getKey() == funct_location ) {
                     std::string description = iD->getValueDescription( e.getKey(), e.getValue() );
                     fprintf(stderr,"NANOS++: (TASK) Finishing %s function location\n", description.c_str() );
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

DECLARE_PLUGIN("intrumentation-print_trace",nanos::ext::InstrumentationPrintTracePlugin);
