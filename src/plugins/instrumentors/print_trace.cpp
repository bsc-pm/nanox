#include "plugin.hpp"
#include "instrumentor.hpp"
#include "system.hpp"

namespace nanos {

class InstrumentorPrintTrace: public Instrumentor 
{
   private:
   public:
      // constructor
      InstrumentorPrintTrace ( ) { }

      // destructor
      ~InstrumentorPrintTrace ( ) { }

      // headers (implemented below)
      // low-level instrumentation interface

      void pushState( int state ){ fprintf(stderr,"pushState\n"); }
      void popState( void ) { fprintf(stderr,"popState\n");  }
      void addEvent() { fprintf(stderr,"addEvent\n");  } 
      void addEventList() { fprintf(stderr,"addEventList\n");  } 

      // high-level events

      void enterRuntime () { fprintf(stderr, "TRACE [%d]: Entering runtime.\n", myThread->getId() ); } 
      void leaveRuntime () { fprintf(stderr, "TRACE [%d]: Leaving runtime.\n", myThread->getId() );  }
      void enterBarrier () { fprintf(stderr, "TRACE [%d]: Entering barrier.\n", myThread->getId() ); }
      void leaveBarrier () { fprintf(stderr, "TRACE [%d]: Leaving barrier.\n", myThread->getId() ); }
      void enterCPU () {}
      void leaveCPU () {}

      void threadIdle() {}

      void taskCreation() {}
      void taskCompletation() {}


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
