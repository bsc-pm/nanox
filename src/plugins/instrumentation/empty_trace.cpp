#include "plugin.hpp"
#include "system.hpp"
#include "instrumentation.hpp"

namespace nanos {

class InstrumentationEmptyTrace: public Instrumentation 
{
#ifndef NANOS_INSTRUMENTATION_ENABLED
   public:
      // constructor
      InstrumentationEmptyTrace( ) : Instrumentation( ) {}
      // destructor
      ~InstrumentationEmptyTrace() {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void ) {}
      void finalize( void ) {}
      void disable( void ) {}
      void enable( void ) {}
      void addResumeTask( WorkDescriptor &w ) {}
      void addSuspendTask( WorkDescriptor &w, bool last ) {}
      void addEventList ( unsigned int count, Event *events ) {}
      void threadStart( BaseThread &thread ) {}
      void threadFinish ( BaseThread &thread ) {}
#else
   public:
      // constructor
      InstrumentationEmptyTrace( ) : Instrumentation( *new InstrumentationContext() ) {}
      // destructor
      ~InstrumentationEmptyTrace () {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void ) {}
      void finalize( void ) {}
      void disable( void ) {}
      void enable( void ) {}
      void addResumeTask( WorkDescriptor &w ) {}
      void addSuspendTask( WorkDescriptor &w, bool last ) {}
      void addEventList ( unsigned int count, Event *events ) {}
      void threadStart( BaseThread &thread ) {}
      void threadFinish ( BaseThread &thread ) {}
#endif
};

namespace ext {

class InstrumentationEmptyTracePlugin : public Plugin {
   public:
      InstrumentationEmptyTracePlugin () : Plugin("Instrumentation which doesn't generate any trace.",1) {}
      ~InstrumentationEmptyTracePlugin () {}

      void config( Config &cfg ) {}

      void init ()
      {
         sys.setInstrumentation( new InstrumentationEmptyTrace() );	
      }
};

} // namespace ext

} // namespace nanos

DECLARE_PLUGIN("instrumentation-empty_trace",nanos::ext::InstrumentationEmptyTracePlugin);
