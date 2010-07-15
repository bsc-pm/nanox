#include "plugin.hpp"
#include "system.hpp"
#include "instrumentor.hpp"

namespace nanos {

class InstrumentationEmptyTrace: public Instrumentation 
{
#ifndef NANOS_INSTRUMENTATION_ENABLED
   public:
      // constructor
      InstrumentationEmptyTrace() : Instrumentation() {}
      // destructor
      ~InstrumentationEmptyTrace() {}

      // low-level instrumentation interface (mandatory functions)
      virtual void initialize( void ) {}
      virtual void finalize( void ) {}
      virtual void addEventList ( unsigned int count, Event *events ) {}
#else
   private:
      InstrumentationContext   _icLocal;

   public:
      // constructor
      InstrumentationEmptyTrace() : Instrumentation(), _icLocal() { _instrumentationContext = &_icLocal; }
      // destructor
      ~InstrumentationEmptyTrace () {}

      // low-level instrumentation interface (mandatory functions)
      virtual void initialize( void ) {}
      virtual void finalize( void ) {}
      virtual void addEventList ( unsigned int count, Event *events ) {}
#endif
};

namespace ext {

class InstrumentorEmptyTracePlugin : public Plugin {
   public:
      InstrumentorEmptyTracePlugin () : Plugin("Instrumentor which doesn't generate any trace.",1) {}
      ~InstrumentorEmptyTracePlugin () {}

      virtual void config( Config &config ) {}

      void init ()
      {
         sys.setInstrumentor( new InstrumentationEmptyTrace() );	
      }
};

} // namespace ext

} // namespace nanos

nanos::ext::InstrumentorEmptyTracePlugin NanosXPlugin;
