#include "plugin.hpp"
#include "instrumentor_decl.hpp"
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

class InstrumentorParaver: public Instrumentor 
{
   private:
       static const unsigned int _eventState    = 9000;   /*<< event coding state changes */
       static const unsigned int _eventPTPStart = 9001;   /*<< event coding state comm start */
       static const unsigned int _eventPTPEnd   = 9002;   /*<< event coding state comm end */
       static const unsigned int _eventBase     = 9900;   /*<< event coding basecode for general events (kvs) */
       unsigned int _states[LAST_EVENT_STATE];            /*<< state vector translator */
       unsigned int _events[LAST_EVENT];                  /*<< event id vector translator */

   public:
      // constructor
      InstrumentorParaver ( )
      {
#if 0
         _states[ERROR]           = 66;
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

         _events[THROW_WD]        = 9910;
         _events[CATCH_WD]        = 9911;
 #endif
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
            p_file << "9    " << _eventState  << "    Change status: " << std::endl;
            p_file << "VALUES" << std::endl;

            // List of states
            p_file << ERROR           << "     ERROR" << std::endl;
            p_file << _states[IDLE]            << "     IDLE" << std::endl;
            p_file << _states[RUNNING]         << "     RUNNING" << std::endl;
            p_file << _states[SYNCHRONIZATION] << "     SYNCHRONIZATION" << std::endl;
            p_file << _states[SCHEDULING]      << "     SCHED-FORK/JOIN" << std::endl;
            p_file << _states[FORK_JOIN]       << "     SCHED-FORK/JOIN" << std::endl;
            p_file << _states[OTHERS]          << "     OTHERS" << std::endl;

            p_file << std::endl;

            // FIXME (141): Generate paraver .pcf file automatically using registerd event types
#if 0
            // List of events enter/leave functions
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    " << _events[IDLE_FUNCTION]  << "    Idle function" << std::endl;
            p_file << "9    " << _events[RUNTIME]        << "    Runtime region" << std::endl;
            p_file << "9    " << _events[CREATE_WD]      << "    Create WD" << std::endl;
            p_file << "9    " << _events[SUBMIT_WD]      << "    Submit WD" << std::endl;
            p_file << "9    " << _events[INLINE_WD]      << "    Inline WD" << std::endl;
            p_file << "9    " << _events[LOCK]           << "    Lock region" << std::endl;
            p_file << "9    " << _events[SINGLE_GUARD]   << "    Single guard region" << std::endl;
            p_file << "9    " << _events[BARRIER]        << "    Barrier region" << std::endl;
            p_file << "9    " << _events[SWITCH]         << "    Switch region" << std::endl;

            p_file << "VALUES" << std::endl;
            p_file << "1      begins" << std::endl;
            p_file << "0      ends"   << std::endl;
#endif

            p_file << std::endl;

            // List of events trhow/catch 
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    " << _events[THROW_WD]  << "    Throwing task:" << std::endl;
            p_file << "9    " << _events[CATCH_WD]  << "    Catching task:" << std::endl;

            p_file.close();
         }
         else std::cout << "Unable to open paraver config file" << std::endl;  
      }

      void addEventList ( unsigned int count, Event *events) 
      {
         int total = 0;
         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            switch ( e.getType() ) {
               case Event::STATE:
                  total++;
                  break;
               case Event::PTP_START:
               case Event::PTP_END:
                  total++;
               case Event::POINT:
               case Event::BURST_START:
               case Event::BURST_END:
                  total += e.getNumKVs();
                  break;
            }
         }

         unsigned int *p_events = (unsigned int *) alloca (total * sizeof (unsigned int));
         unsigned int *p_values = (unsigned int *) alloca (total * sizeof (unsigned int));

         int j = 0;
         Event::ConstKVList kvs = NULL;


         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            int localBase=0;
            switch ( e.getType() ) {
               case Event::STATE:
                  p_events[j] = _eventState;
                  p_values[j++] = e.getState();
                  break;
               case Event::PTP_START:
               case Event::PTP_END:
                  p_events[j] = ( e.getType() == Event::PTP_START ) ? _eventPTPStart : _eventPTPEnd;
	          p_values[j++] = e.getDomain()+e.getId();
                  localBase = 1000;
               case Event::POINT:
               case Event::BURST_START:
                  kvs = e.getKVs();
                  for ( unsigned int kv = 0 ; kv < e.getNumKVs() ; kv++,kvs++ ) {
                     p_events[j] = localBase + _eventBase + kvs->first;
                     p_values[j++] = kvs->second;
                  }
                  break;
               case Event::BURST_END:
                  kvs = e.getKVs();
                  for ( unsigned int kv = 0 ; kv < e.getNumKVs() ; kv++,kvs++ ) {
                     p_events[j] = _eventBase +  kvs->first;
                     p_values[j++] = 0; // end
                  }
                  break;
            }
         }

         int rmValues = 0;
         for ( int i = 0; i < total; i++ )
            for ( int j = i+1; j < total; j++ )
               if ( p_events[i] == p_events[j] )
               {
                  p_events[i] = 0;
                  rmValues++;
               }

         total -= rmValues;

         for ( int j = 0, i = 0; i < total; i++ )
         {
            while ( p_events[j] == 0 ) j++;
            p_events[i] = p_events[j];
            p_values[i] = p_values[j++];
         }

         OMPItrace_neventandcounters(total , p_events, p_values);
          
      }
};


namespace ext {

class InstrumentorParaverPlugin : public Plugin {
   public:
      InstrumentorParaverPlugin () : Plugin("Instrumentor which generates a paraver trace.",1) {}
      ~InstrumentorParaverPlugin () {}

      virtual void config( Config &config ) {}

      void init ()
      {
         sys.setInstrumentor( new InstrumentorParaver() );
      }
};

} // namespace ext
} // namespace nanos

nanos::ext::InstrumentorParaverPlugin NanosXPlugin;
