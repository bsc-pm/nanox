#include "plugin.hpp"
#include "instrumentor.hpp"
#include "system.hpp"

namespace nanos {

#define INSTRUMENTOR_MAX_STATES 10
#define INSTRUMENTOR_MAX_EVENTS 10

class InstrumentorPrintTrace: public Instrumentor 
{
   private:
       unsigned int _states[INSTRUMENTOR_MAX_STATES];  /*<< state vector translator */
       unsigned int _events[INSTRUMENTOR_MAX_EVENTS];  /*<< event id vector translator */

   public:
      // constructor
      InstrumentorPrintTrace ( )
      {
         _states[IDLE]      = 80000000;
         _states[RUNNING]   = 80000001;

         _events[CREATE_WD] = 80000002;

      }

      // destructor
      ~InstrumentorPrintTrace ( ) { }

      // headers (implemented below)
      // low-level instrumentation interface

      void initialize ( void ) { }
      void finalize ( void ) { } 
      void changeStateEventList ( nanos_state_t state, unsigned int count, nanos_event_t *events ) { }
      void addEventList ( unsigned int count, nanos_event_t *events ) { }

      // high-level events

      void enterCreateWD () { fprintf(stderr, "TRACE [%d]: Enter create WD (%d).\n", myThread->getId(), _states[CREATE_WD] ); }
      void leaveCreateWD () { fprintf(stderr, "TRACE [%d]: Leave create WD (%d).\n", myThread->getId(), _states[CREATE_WD] ); }

};


namespace ext {

class InstrumentorPrintTracePlugin : public Plugin {
   public:
      InstrumentorPrintTracePlugin () : Plugin("Instrumentor which print the trace to std out.",1) {}
      ~InstrumentorPrintTracePlugin () {}

      void init ()
      {
         sys.setInstrumentor( new InstrumentorPrintTrace() );	
      }
};

} // namespace ext
} // namespace nanos

nanos::ext::InstrumentorPrintTracePlugin NanosXPlugin;
