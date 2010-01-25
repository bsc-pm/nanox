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

      // headers (implemented below)
      // low-level instrumentation interface

      void pushState( int state ){ }
      void popState( void ) { }
      void addEvent() { } 
      void addEventList() { } 

      // high-level events

      void enterRuntime () { } 
      void leaveRuntime () { }

      void enterCPU () {}
      void leaveCPU () {}

      void threadIdle() {}

      void taskCreation() {}
      void taskCompletation() {}

      void enterBarrier() {}
      void exitBarrier() {}

};


namespace ext {

class InstrumentorEmptyTracePlugin : public Plugin {
   public:
      InstrumentorEmptyTracePlugin () : Plugin("Instrumentor which doesn't generate any trace.",1) {}
      ~InstrumentorEmptyTracePlugin () {}

      void init ()
      {
         sys.setInstrumentor( new InstrumentorEmptyTrace() );	
      }
};

} // namespace ext
} // namespace nanos

nanos::ext::InstrumentorEmptyTracePlugin NanosXPlugin;
