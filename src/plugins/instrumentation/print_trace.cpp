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
         fprintf(stderr,"NANOS++: Resumed task %d in thread %d\n",w.getId(), myThread->getId());
      }
      void addSuspendTask( WorkDescriptor &w, bool last ) {
         fprintf(stderr,"NANOS++: %s task %d in thread %d\n",last?"Finished":"Suspended",w.getId(), myThread->getId());
      }
      void addEventList ( unsigned int count, Event *events )
      {
         // Getting predefined key's
         InstrumentationDictionary *iD = sys.getInstrumentation()->getInstrumentationDictionary();
         nanos_event_key_t create_task = iD->getEventKey("create-wd-ptr"); 
         nanos_event_key_t funct_location = iD->getEventKey("user-funct-location"); 
         nanos_event_key_t dependence  = iD->getEventKey("dependence");
         nanos_event_key_t dep_address  = iD->getEventKey("dep-address");

         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            int64_t value = e.getValue();
            switch ( e.getType() ) {
               case NANOS_STATE_START:
                  break;
               case NANOS_STATE_END:
                  break;
               case NANOS_SUBSTATE_START:
                  break;
               case NANOS_SUBSTATE_END:
                  break;
               case NANOS_PTP_START:
                  break;
               case NANOS_PTP_END:
                  break;
               case NANOS_POINT:
                  if ( e.getKey() == create_task ) {
                     WorkDescriptor *wd = (WorkDescriptor *) value;
                     int64_t wd_id = wd->getId();
                     int64_t funct_id = (int64_t) ((ext::SMPDD &) (wd->getActiveDevice ())).getWorkFct ();
                     fprintf(stderr,"NANOS++: Executing %"PRId64" function within task %"PRId64" in thread %d\n",funct_id,wd_id, myThread->getId());
                  }
                  if ( (nanos_event_key_t)(events[i]).getKey() == dependence ) {
                     nanos_event_value_t dependence_value = (events[i]).getValue();
                     int sender_id = (int) ( dependence_value >> 32 );
                     int receiver_id = (int) ( dependence_value & 0xFFFFFFFFFF );

                     while ( (i < count) && ((nanos_event_key_t)(events[i]).getKey() != dep_address) ) i++;

                     void * address_id = 0;
                     if ( i < count ) address_id = (void *) ((events[i]).getValue());

                     fprintf(stderr,"NANOS++: Adding dependence %d->%d (related data address %p)\n",sender_id,receiver_id, address_id );
                  }

                  break;
               case NANOS_BURST_START:
                  if ( e.getKey() == funct_location ) {
                     std::string description = iD->getValueDescription( e.getKey(), e.getValue() );
                     fprintf(stderr,"NANOS++: Executing %s function location\n", description.c_str() );
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
#if 0

void addEventList ( unsigned int count, Event *events )
      {
         InstrumentationDictionary *iD = sys.getInstrumentation()->getInstrumentationDictionary();

         int64_t AYU_data[3];

         for (unsigned int i = 0; i < count; i++)
         {
            switch ( (events[i]).getType() ) {
               case NANOS_STATE_START:
               case NANOS_STATE_END:
               case NANOS_SUBSTATE_START:
               case NANOS_SUBSTATE_END:
               case NANOS_PTP_START:
               case NANOS_PTP_END:
                  break;
               case NANOS_POINT:
                  break;
               case NANOS_BURST_START:
               case NANOS_BURST_END:
               default: break;
            }
         }
      }


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
