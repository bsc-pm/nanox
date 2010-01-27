#include "plugin.hpp"
#include "instrumentor.hpp"
#include "system.hpp"
#include <mpitrace_user_events.h>

namespace nanos {

class InstrumentorParaver: public Instrumentor 
{
   private:
   public:
      // constructor
      InstrumentorParaver ( ) { }

      // destructor
      ~InstrumentorParaver ( ) { }

      // headers (implemented below)
      // low-level instrumentation interface

      void pushStateEventList ( nanos_state_t state, int count, nanos_event_t *events ) { }
      void popStateEventList ( int count, nanos_event_t *events ) { }
      void addEventList ( int count, nanos_event_t *event) { }

};


namespace ext {

class InstrumentorParaverPlugin : public Plugin {
   public:
      InstrumentorParaverPlugin () : Plugin("Instrumentor which generates a paraver trace.",1) {}
      ~InstrumentorParaverPlugin () {}

      void init ()
      {
         sys.setInstrumentor( new InstrumentorParaver() );
      }
};

} // namespace ext
} // namespace nanos

nanos::ext::InstrumentorParaverPlugin NanosXPlugin;
