#include "plugin.hpp"
#include "instrumentor.hpp"
#include "system.hpp"

namespace nanos {

class InstrumentorEmptyTrace: public Instrumentor 
{
   private:
   public:
      // constructor
      InstrumentorEmptyTrace ( ) { }

      // destructor
      ~InstrumentorEmptyTrace ( ) { }

      // low-level instrumentation interface (mandatory functions)

      void initialize ( void ) { }
      void finalize ( void ) { }
      void pushStateEventList ( nanos_state_t state, unsigned int count, nanos_event_t *event ) { }
      void popStateEventList ( nanos_state_t state, unsigned int count, nanos_event_t *event ) { }
      void addEventList ( unsigned int count, nanos_event_t *event ) { } 

      // mid-level interface

      void pushState ( nanos_state_t state ) { }
      void popState( nanos_state_t state ) { }
      void pushStateEvent ( nanos_state_t state, nanos_event_t event) { }
      void popStateEvent( nanos_state_t state, nanos_event_t event ) { }
      void addEvent( nanos_event_t event ) { }

      // high-level events

      void enterCreateWD() { }
      void leaveCreateWD() { }
};


namespace ext {

class InstrumentorEmptyTracePlugin : public Plugin {
   public:
      InstrumentorEmptyTracePlugin () : Plugin("Instrumentor which doesn't generate any trace.",1) {}
      ~InstrumentorEmptyTracePlugin () {}

      virtual void config( Config &config ) {}

      void init ()
      {
         sys.setInstrumentor( new InstrumentorEmptyTrace() );	
      }
};

} // namespace ext
} // namespace nanos

nanos::ext::InstrumentorEmptyTracePlugin NanosXPlugin;
