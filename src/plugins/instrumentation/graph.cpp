#include "plugin.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"

namespace nanos {

class InstrumentationGraphInstrumentation: public Instrumentation 
{
#ifndef NANOS_INSTRUMENTATION_ENABLED
   public:
      // constructor
      InstrumentationGraphInstrumentation() : Instrumentation() {}
      // destructor
      ~InstrumentationGraphInstrumentation() {}

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
      InstrumentationGraphInstrumentation() : Instrumentation( *new InstrumentationContext() ) {}
      // destructor
      ~InstrumentationGraphInstrumentation ( ) {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void )
      {
         fprintf(stderr,"digraph {\n");
      }
      void finalize( void )
      {
         fprintf(stderr,"}\n");
         // Execute: dot -Tpng graph.dot -ograph.png
      }
      void disable( void ) {}
      void enable( void ) {}
      void addResumeTask( WorkDescriptor &w ) {}
      void addSuspendTask( WorkDescriptor &w, bool last ) {}
      void addEventList ( unsigned int count, Event *events )
      {
         unsigned int i;
         for( i=0; i<count; i++) {
            Event &e = events[i];
            if ( e.getKey() == 36 ) {
               unsigned sender = (e.getValue() >> 32) & 0xFFFFFFFF;
               unsigned receiver = e.getValue() & 0xFFFFFFFF;
               fprintf(stderr,"  %d -> %d;\n",sender,receiver);
            }
         }
      }
      void threadStart( BaseThread &thread ) {}
      void threadFinish ( BaseThread &thread ) {}
#endif

};

namespace ext {

class InstrumentationGraphInstrumentationPlugin : public Plugin {
   public:
      InstrumentationGraphInstrumentationPlugin () : Plugin("Instrumentation which print the trace to std out.",1) {}
      ~InstrumentationGraphInstrumentationPlugin () {}

      void config( Config &cfg ) {}

      void init ()
      {
         sys.setInstrumentation( new InstrumentationGraphInstrumentation() );	
      }
};

} // namespace ext

} // namespace nanos

DECLARE_PLUGIN("intrumentation-print_trace",nanos::ext::InstrumentationGraphInstrumentationPlugin);
