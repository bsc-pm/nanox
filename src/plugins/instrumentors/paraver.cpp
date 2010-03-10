#include "plugin.hpp"
#include "system.hpp"
#include "instrumentor.hpp"
#include <mpitrace_user_events.h>
#include "debug.hpp"
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <alloca.h>
#include <stdlib.h>

namespace nanos {

   const unsigned int _eventState      = 9000;   /*<< event coding state changes */
   const unsigned int _eventPtPStart   = 9001;   /*<< event coding comm start */
   const unsigned int _eventPtPEnd     = 9002;   /*<< event coding comm end */

   class InstrumentorParaver: public Instrumentor 
   {
#if defined INSTRUMENTATION_ENABLED
   private:
      unsigned int _eventBase[EVENT_TYPES];
   public:
      // constructor
      InstrumentorParaver ( )
      {
         _eventBase[STATE]       = 0;
         _eventBase[BURST_START] = 9200;
         _eventBase[BURST_END]   = 9200;
         _eventBase[PTP_START]   = 9400;
         _eventBase[PTP_END]     = 9400;
         _eventBase[POINT]       = 9600;
      }

      // destructor
      ~InstrumentorParaver ( ) { }

      void mergeParaverTraceFiles ()
      {
         char str[255];
         int status, options = 0;
         pid_t pid;

         // Merging trace files
         strcpy(str, MPITRACE_BIN);
         strcat(str, "/mpi2prv");
         pid = fork();
         if ( pid == (pid_t) 0 ) execl ( str, "mpi2prv", "-f", "TRACE.mpits", (char *) NULL);
         else waitpid( pid, &status, options);

         // Deleting temporary files
         std::fstream p_file;
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
      }

      void createParaverConfigFile()
      {
         // Writing paraver config 
         std::fstream p_file;
         p_file.open ("MPITRACE_Paraver_Trace.pcf", std::ios::out | std::ios::app);
         if (p_file.is_open())
         {
            /* Event: State */
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    " << _eventState  << "    Change status: " << std::endl;
            p_file << "VALUES" << std::endl;
            p_file << NOT_TRACED       << "     NOT TRACED" << std::endl;
            p_file << ERROR            << "     ERROR" << std::endl;
            p_file << IDLE             << "     IDLE" << std::endl;
            p_file << RUNTIME          << "     RUNTIME" << std::endl;
            p_file << RUNNING          << "     RUNNING" << std::endl;
            p_file << SYNCHRONIZATION  << "     SYNCHRONIZATION" << std::endl;
            p_file << SCHEDULING       << "     SCHED-FORK/JOIN" << std::endl;
            p_file << FORK_JOIN        << "     SCHED-FORK/JOIN" << std::endl;
            p_file << std::endl;

            /* Event: PtPStart */
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    " << _eventPtPStart  << "    Point-to-point origin: " << std::endl;
            p_file << std::endl;

            /* Event: PtPEnd */
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    " << _eventPtPEnd    << "    Point-to-point destination: " << std::endl;
            p_file << std::endl;

            /* Event: Burst (NANOS_API) */
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    " << _eventBase[BURST_START]+NANOS_API  << "     API Fucntion: " << std::endl;
            p_file << "VALUES" << std::endl;
            p_file << NOT_IN_NANOS_API     << "     not in nanos api" << std::endl;
            p_file << CURRENT_WD           << "     current wd" << std::endl;
            p_file << GET_WD_ID            << "     get wd id" << std::endl;
            p_file << CREATE_WD            << "     create wd" << std::endl;
            p_file << SUBMIT_WD            << "     submit wd" << std::endl;
            p_file << CREATE_WD_AND_RUN    << "     create wd and run" << std::endl;
            p_file << SET_INTERNAL_WD_DATA << "     set internal wd data" << std::endl;
            p_file << GET_INTERNAL_WD_DATA << "     get internal wd data" << std::endl;
            p_file << YIELD                << "     yield" << std::endl;
            p_file << WG_WAIT_COMPLETATION << "     wg wait completation" << std::endl;
            p_file << SYNC_COND            << "     sync cond" << std::endl;
            p_file << WAIT_ON              << "     wait on" << std::endl;
            p_file << LOCK                 << "     lock function" << std::endl;
            p_file << SINGLE_GUARD         << "     single guard" << std::endl;
            p_file << TEAM_BARRIER         << "     team barrier" << std::endl;
            p_file << FIND_SLICER          << "     find slicer" << std::endl;
            p_file << std::endl;

            /* Event: Burst (WD_ID) */
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    " << _eventBase[BURST_START]+WD_ID  << "     Work Descriptor, id: " << std::endl;
            p_file << std::endl;

            p_file.close();
         }
         else std::cout << "Unable to open paraver config file" << std::endl;  
      }

      void initialize ( void )
      {
         char *mpi_trace_on= new char[255];
         strcpy(mpi_trace_on, "MPITRACE_ON=1");
         putenv (mpi_trace_on);
         OMPItrace_init();
      }

      void finalize ( void )
      {
         OMPItrace_fini();
         mergeParaverTraceFiles();
         createParaverConfigFile();
      }

      void addEventList ( unsigned int count, Event *events) 
      {
         unsigned int total = 0;
         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            switch ( e.getType() ) {
               case STATE:
                  total++;
                  break;
               case PTP_START:
               case PTP_END:
                  total++;
                  // continue...
               case POINT:
               case BURST_START:
               case BURST_END:
                  total += e.getNumKVs();
                  break;
               default: break;
            }
         }

         unsigned int *p_events = (unsigned int *) alloca (total * sizeof (unsigned int));
         unsigned int *p_values = (unsigned int *) alloca (total * sizeof (unsigned int));
        

         int j = 0;
         Event::ConstKVList kvs = NULL;


         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            unsigned int type = e.getType();
            switch ( type ) {
               case STATE:
                  p_events[j] = _eventState;
                  p_values[j++] = e.getState();
                  break;
               case PTP_START:
               case PTP_END:
                  /* Creating PtP event */
		  if ( type == PTP_START ) p_events[j] = _eventPtPStart;
		  else p_events[j] = _eventPtPEnd;
	          p_values[j++] = e.getDomain() + e.getId();
                  // continue...
               case POINT:
               case BURST_START:
                  kvs = e.getKVs();
                  for ( unsigned int kv = 0 ; kv < e.getNumKVs() ; kv++,kvs++ ) {
                     p_events[j] = _eventBase[type] + kvs->first;
                     p_values[j++] = kvs->second;
                  }
                  break;
               case BURST_END:
                  kvs = e.getKVs();
                  for ( unsigned int kv = 0 ; kv < e.getNumKVs() ; kv++,kvs++ ) {
                     p_events[j] = _eventBase[type] +  kvs->first;
                     p_values[j++] = 0; // end
                  }
                  break;
               default: break;
            }
         }

         int rmValues = 0;
         for ( unsigned int i = 0; i < total; i++ )
            for ( unsigned int j = i+1; j < total; j++ )
               if ( p_events[i] == p_events[j] )
               {
                  p_events[i] = 0;
                  rmValues++;
               }

         total -= rmValues;

         for ( unsigned int j = 0, i = 0; i < total; i++ )
         {
            while ( p_events[j] == 0 ) j++;
            p_events[i] = p_events[j];
            p_values[i] = p_values[j++];
         }

         OMPItrace_neventandcounters(total , p_events, p_values);
          
      }
#endif
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
