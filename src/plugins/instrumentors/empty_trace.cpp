#include "plugin.hpp"
#include "system.hpp"
#include "instrumentor.hpp"

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

      virtual void initialize( void ) {}
      virtual void finalize( void ) {}
      virtual void addEventList ( unsigned int count, Event *events ) {}
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
