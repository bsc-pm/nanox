#include "plugin.hpp"
#include "instrumentor.hpp"
#include "system.hpp"
#include <mpitrace_user_events.h>
#include "debug.hpp"
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <iostream>
#include <fstream>

namespace nanos {

extern "C" {

   void OMPItrace_neventandcounters (unsigned int count, unsigned int *types, unsigned int *values);

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
       static const unsigned int _eventState = 9000; /*<< event coding state changes */
       unsigned int _states[INSTRUMENTOR_MAX_STATES];    /*<< state vector translator */
       unsigned int _events[INSTRUMENTOR_MAX_EVENTS];    /*<< event id vector translator */

   public:
      // constructor
      InstrumentorParaver ( )
      {
         _states[IDLE]            = 0;
         _states[RUNNING]         = 1;
         _states[SYNCHRONIZATION] = 5;
         _states[SCHEDULING]      = 7;
         _states[FORK_JOIN]       = 7;
         _states[OTHERS]          = 15;

         _events[IDLE_FUNCTION]   = 9901;
         _events[RUNTIME]         = 9902;
         _events[CREATE_WD]       = 9903;
         _events[SUBMIT_WD]       = 9904;
         _events[INLINE_WD]       = 9905;
         _events[LOCK]            = 9906;
         _events[SINGLE_GUARD]    = 9907;
         _events[BARRIER]         = 9908;
         _events[SWITCH]          = 9909;
 
      }

      // destructor
      ~InstrumentorParaver ( ) { }

      // headers (implemented below)
      // low-level instrumentation interface

      void initialize ( void )
      {
         char *mpi_trace_on= new char[255];
         strcpy(mpi_trace_on, "MPITRACE_ON=1");
         putenv (mpi_trace_on);
         OMPItrace_init();
      }

      void finalize ( void )
      {
         char str[255];
         int status, options = 0;
         pid_t pid;
         std::fstream p_file;

         OMPItrace_fini();

         // Merging trace files
         strcpy(str, MPITRACE_BIN);
         strcat(str, "/mpi2prv");
         pid = fork();
         if ( pid == (pid_t) 0 ) execl ( str, "mpi2prv", "-f", "TRACE.mpits", (char *) NULL);
         else waitpid( pid, &status, options);

         // Deleting temporary files
         p_file.open("TRACE.mpits");
         if (p_file.is_open())
         {
            while (!p_file.eof() )
            {
               p_file.getline (str, 255);
               if ( strlen(str) > 0 )
               {
                  for (unsigned int i = 0; i < strlen(str); i++) if ( str[i] == ' ' ) str[i] = 0x0;
                  if ( remove(str) != 0 ) std::cout << "Unable to delete file" << str << std::endl;
               }
            }
            p_file.close();
         }
         else std::cout << "Unable to open file" << std::endl;  
         if ( remove("TRACE.mpits") != 0 ) std::cout << "Unable to delete TRACE.mpits file" << std::endl;

         // Writing paraver config 
         p_file.open ("MPITRACE_Paraver_Trace.pcf", std::ios::out | std::ios::app);
         if (p_file.is_open())
         {
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    9000    Change status" << std::endl;
            p_file << "VALUES" << std::endl;
            p_file << "0      Pop" << std::endl;

            p_file << std::endl;

            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    9900    Unspecified region" << std::endl;
            p_file << "9    9901    Idle function" << std::endl;
            p_file << "9    9902    Runtime region" << std::endl;
            p_file << "9    9903    Create WD" << std::endl;
            p_file << "9    9904    Submit WD" << std::endl;
            p_file << "9    9905    Inline WD" << std::endl;
            p_file << "9    9906    Lock region" << std::endl;
            p_file << "9    9907    Single guard region" << std::endl;
            p_file << "9    9908    Barrier region" << std::endl;
            p_file << "VALUES" << std::endl;
            p_file << "1      Begin" << std::endl;
            p_file << "0      End" << std::endl;

            p_file.close();
         }
         else std::cout << "Unable to open paraver config file" << std::endl;  
      }

      void pushStateEventList ( nanos_state_t state, unsigned int count, nanos_event_t *events )
      {
         unsigned int *p_events = (unsigned int *) alloca ((count+1) * sizeof (unsigned int));
         unsigned int *p_values = (unsigned int *) alloca ((count+1) * sizeof (unsigned int));

         p_events[0] = _eventState;
         p_values[0] = _states[state];

         for (unsigned int i = 0; i < count; i++)
         {
            p_events[i+1] = _events[events[i].id];
            p_values[i+1] = events[i].value;
         }

         OMPItrace_neventandcounters(count+1, p_events, p_values);
      }
      void popStateEventList ( nanos_state_t state, unsigned int count, nanos_event_t *events )
      {
         unsigned int *p_events = (unsigned int *) alloca ((count+1) * sizeof (unsigned int));
         unsigned int *p_values = (unsigned int *) alloca ((count+1) * sizeof (unsigned int));

         p_events[0] = _eventState;
         p_values[0] = 0;

         for (unsigned int i = 0; i < count; i++)
         {
            p_events[i+1] = _events[events[i].id];
            p_values[i+1] = events[i].value;
         }

         OMPItrace_neventandcounters(count+1, p_events, p_values);
      }
      void addEventList ( unsigned int count, nanos_event_t *events) 
      {
         unsigned int *p_events = (unsigned int *) alloca (count * sizeof (unsigned int));
         unsigned int *p_values = (unsigned int *) alloca (count * sizeof (unsigned int));

         for (unsigned int i = 0; i < count; i++)
         {
            p_events[i] = _events[events[i].id];
            p_values[i] = events[i].value;
         }

         OMPItrace_neventandcounters(count, p_events, p_values);
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
