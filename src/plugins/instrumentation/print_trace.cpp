#include "plugin.hpp"
#include "system.hpp"
#include "instrumentor.hpp"
#include "instrumentorcontext_decl.hpp"

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
      virtual void initialize( void ) {}
      virtual void finalize( void ) {}
      virtual void addEventList ( unsigned int count, Event *events ) {}
#else
   private:
      InstrumentationContext   _icLocal;

   public:
      // constructor
      InstrumentationPrintTrace() : Instrumentation(), _icLocal() { _instrumentationContext = &_icLocal; }
      // destructor
      ~InstrumentationPrintTrace ( ) {}

      // low-level instrumentation interface (mandatory functions)
      virtual void initialize( void ) {}
      virtual void finalize( void ) {}
      virtual void addEventList ( unsigned int count, Event *events ) {}
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
         sys.setInstrumentor( new InstrumentationPrintTrace() );	
      }
};

} // namespace ext

} // namespace nanos

nanos::ext::InstrumentorPrintTracePlugin NanosXPlugin;

