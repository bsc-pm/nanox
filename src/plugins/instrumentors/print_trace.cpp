#include "plugin.hpp"
#include "system.hpp"
#include "instrumentor.hpp"

namespace nanos {

class InstrumentorPrintTrace: public Instrumentor 
{
#if defined INSTRUMENTATION_ENABLED
   private:

   public:
      // constructor
      InstrumentorPrintTrace ( )
      {
      }

      // destructor
      ~InstrumentorPrintTrace ( ) { }

      // low-level instrumentation interface (mandatory functions)

      virtual void initialize( void ) {}
      virtual void finalize( void ) {}
      virtual void addEventList ( unsigned int count, Event *events ) {}

      // high-level events

      virtual void enterRuntimeAPI ( nanos_event_api_t function, nanos_event_state_t state = RUNTIME ) {}
      virtual void leaveRuntimeAPI ( ) {}
      virtual void enterIdle ( ) {}
      virtual void leaveIdle ( ) {}
      virtual void wdSwitch( WorkDescriptor* oldWD, WorkDescriptor* newWD ) {}
      virtual void wdExit( WorkDescriptor* oldWD, WorkDescriptor* newWD ) {}
#endif
};


namespace ext {

class InstrumentorPrintTracePlugin : public Plugin {
   public:
      InstrumentorPrintTracePlugin () : Plugin("Instrumentor which print the trace to std out.",1) {}
      ~InstrumentorPrintTracePlugin () {}

      virtual void config( Config &config ) {}

      void init ()
      {
         sys.setInstrumentor( new InstrumentorPrintTrace() );	
      }
};

} // namespace ext
} // namespace nanos

nanos::ext::InstrumentorPrintTracePlugin NanosXPlugin;
