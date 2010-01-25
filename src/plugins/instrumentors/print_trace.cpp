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

      void enterRuntime () { fprintf(stderr, "TRACE: Entering runtime.\n"); } 
      void leaveRuntime () { fprintf(stderr, "TRACE: Leaving runtime.\n");  }
      void enterBarrier () { fprintf(stderr, "TRACE: Entering barrier.\n"); }
      void leaveBarrier () { fprintf(stderr, "TRACE: Leaving barrier.\n"); }

#if 0

      void enterCPU () {}
      void leaveCPU () {}

      void threadIdle() {}

      void taskCreation() {}
      void taskCompletation() {}

#endif

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
