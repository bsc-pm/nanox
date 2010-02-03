#include "plugin.hpp"
#include "instrumentor.hpp"
#include "system.hpp"
#include <mpitrace_user_events.h>
#include "debug.hpp"
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>

namespace nanos {

extern "C" {

   unsigned int nanos_ompitrace_get_max_threads ( void )
   {
      return sys.getNumPEs();
   }

   unsigned int nanos_ompitrace_get_thread_num ( void )
   { 
      return myThread->getId(); 
   }

}

#define INSTRUMENTOR_MAX_STATES 10
#define INSTRUMENTOR_MAX_EVENTS 10

class InstrumentorParaver: public Instrumentor 
{
   private:
       static const unsigned int _eventState = 90000000; /*<< event coding state changes */
       unsigned int _states[INSTRUMENTOR_MAX_STATES];    /*<< state vector translator */
       unsigned int _events[INSTRUMENTOR_MAX_EVENTS];    /*<< event id vector translator */

   public:
      // constructor
      InstrumentorParaver ( )
      {
         _states[IDLE]      = 1000;
         _states[RUN]       = 1001;
         _states[RUNTIME]   = 1002;
         _states[CREATE_WD] = 1004;
         _states[SUBMIT_WD] = 1005;
         _states[INLINE_WD] = 1006;

      }

      // destructor
      ~InstrumentorParaver ( ) { }

      // headers (implemented below)
      // low-level instrumentation interface

      void initialize ( void )
      {
         putenv ("MPITRACE_ON=1");
         OMPItrace_init();
      }

      void finalize ( void )
      {
         char str[255];
         int status, options = 0;
         pid_t pid;

         OMPItrace_fini();

         strcpy(str, MPITRACE_BIN);
         strcat(str, "/mpi2prv");
         pid = fork();
         if ( pid == (pid_t) 0 ) execl ( str, "mpi2prv", "-f", "TRACE.mpits", (char *) NULL);
         else waitpid( pid, &status, options);
      }

      void pushStateEventList ( nanos_state_t state, int count, nanos_event_t *events )
      {
         unsigned int *p_events = (unsigned int *) alloca (count * sizeof (unsigned int));
         unsigned int *p_values = (unsigned int *) alloca (count * sizeof (unsigned int));

         OMPItrace_eventandcounters( _eventState, _states[state] );

         for (int i = 0; i < count; i++)
         {
            p_events[i] = events[i].id;
            p_values[i] = events[i].value;
         }
         //OMPItrace_Neventandcounters(count, p_events, p_values);
      }
      void popStateEventList ( int count, nanos_event_t *events )
      {
         unsigned int *p_events = (unsigned int *) alloca (count * sizeof (unsigned int));
         unsigned int *p_values = (unsigned int *) alloca (count * sizeof (unsigned int));

         OMPItrace_eventandcounters( _eventState, 0 );

         for (int i = 0; i < count; i++)
         {
            p_events[i] = events[i].id;
            p_values[i] = events[i].value;
         }
         //OMPItrace_Neventandcounters(count, p_events, p_values);
      }
      void addEventList ( int count, nanos_event_t *events) 
      {
         unsigned int *p_events = (unsigned int *) alloca (count * sizeof (unsigned int));
         unsigned int *p_values = (unsigned int *) alloca (count * sizeof (unsigned int));

         for (int i = 0; i < count; i++)
         {
            p_events[i] = events[i].id;
            p_values[i] = events[i].value;
         }
         //OMPItrace_Neventandcounters(count, p_events, p_values);
      }
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
