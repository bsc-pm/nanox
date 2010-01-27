#include "plugin.hpp"
#include "instrumentor.hpp"
#include "system.hpp"

namespace nanos {

class InstrumentorPrintTrace: public Instrumentor 
{
   private:
   public:
      // constructor
      InstrumentorPrintTrace ( )
      {
         _states[IDLE]      = 80000000;
         _states[RUN]       = 80000001;
         _states[CREATE_WD] = 80000002;
      }

      // destructor
      ~InstrumentorPrintTrace ( ) { }

      // headers (implemented below)
      // low-level instrumentation interface

      void pushStateEventList ( nanos_state_t state, int count, nanos_event_t *events ) { }
      void popStateEventList ( int count, nanos_event_t *events ) { }
      void addEventList ( int count, nanos_event_t *events ) { }

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
