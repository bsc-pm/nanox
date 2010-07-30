#include "plugin.hpp"
#include "system.hpp"
#include "instrumentor.hpp"
#include "instrumentorcontext_decl.hpp"
#include <extrae_types.h>
#include <mpitrace_user_events.h>
#include "debug.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <alloca.h>
#include <stdlib.h>
#include <libgen.h>
#include "os.hpp"

namespace nanos {

   const unsigned int _eventState      = 9000000;   /*<< event coding state changes */
   const unsigned int _eventPtPStart   = 9000001;   /*<< event coding comm start */
   const unsigned int _eventPtPEnd     = 9000002;   /*<< event coding comm end */
   const unsigned int _eventSubState   = 9000004;   /*<< event coding sub-state changes */
   const unsigned int _eventBase       = 9200000;   /*<< event base (used in key/value pairs) */

class InstrumentationExtrae: public Instrumentation 
{
#ifndef NANOS_INSTRUMENTATION_ENABLED
   public:
      // constructor
      InstrumentationExtrae(): Instrumentation() {}
      // destructor
      ~InstrumentationExtrae() {}

      // low-level instrumentation interface (mandatory functions)
      virtual void initialize( void ) {}
      virtual void finalize( void ) {}
      virtual void addEventList ( unsigned int count, Event *events ) {}
#else
   private:
      InstrumentationContextStackedStatesAndBursts   _icLocal;
   public:
      // constructor
      InstrumentationExtrae ( ) : Instrumentation(), _icLocal() { _instrumentationContext = &_icLocal; }
      // destructor
      ~InstrumentationExtrae ( ) { }

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
         p_file.open ( "MPITRACE_Paraver_Trace.pcf", std::ios::out | std::ios::app);
         if (p_file.is_open())
         {
            /* Event: State */
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    " << _eventState  << "    Thread state: " << std::endl;
            p_file << "VALUES" << std::endl;
            p_file << NOT_CREATED      << "     NOT CREATED" << std::endl;
            p_file << NOT_TRACED       << "     NOT TRACED" << std::endl;
            p_file << STARTUP          << "     STARTUP" << std::endl;
            p_file << SHUTDOWN         << "     SHUTDOWN" << std::endl;
            p_file << ERROR            << "     ERROR" << std::endl;
            p_file << IDLE             << "     IDLE" << std::endl;
            p_file << RUNTIME          << "     RUNTIME" << std::endl;
            p_file << RUNNING          << "     RUNNING" << std::endl;
            p_file << SYNCHRONIZATION  << "     SYNCHRONIZATION" << std::endl;
            p_file << SCHEDULING       << "     SCHEDULING" << std::endl;
            p_file << CREATION         << "     CREATION" << std::endl;
            p_file << MEM_TRANSFER     << "     DATA TRANSFER" << std::endl;
            p_file << CACHE            << "     CACHE ALLOC/FREE" << std::endl;
            p_file << std::endl;

            /* Event: PtPStart main event */
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    " << _eventPtPStart  << "    Point-to-point origin: " << std::endl;
            p_file << std::endl;

            /* Event: PtPEnd main event */
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    " << _eventPtPEnd    << "    Point-to-point destination: " << std::endl;
            p_file << std::endl;

            /* Event: Sub-state (key == state)  */
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    " << _eventSubState  << "    Thread sub-state: " << std::endl;
            p_file << "VALUES" << std::endl;
            p_file << NOT_CREATED      << "     NOT_CREATED" << std::endl;
            p_file << NOT_TRACED       << "     NOT TRACED" << std::endl;
            p_file << STARTUP          << "     STARTUP" << std::endl;
            p_file << SHUTDOWN         << "     SHUTDOWN" << std::endl;
            p_file << ERROR            << "     ERROR" << std::endl;
            p_file << IDLE             << "     IDLE" << std::endl;
            p_file << RUNTIME          << "     RUNTIME" << std::endl;
            p_file << RUNNING          << "     RUNNING" << std::endl;
            p_file << SYNCHRONIZATION  << "     SYNCHRONIZATION" << std::endl;
            p_file << SCHEDULING       << "     SCHEDULING" << std::endl;
            p_file << CREATION         << "     CREATION" << std::endl;
            p_file << MEM_TRANSFER     << "     DATA TRANSFER" << std::endl;
            p_file << CACHE            << "     CACHE ALLOC/FREE" << std::endl;
            p_file << std::endl;

            /* Getting Instrumentor Dictionary */
            InstrumentationDictionary::ConstKeyMapIterator itK;
            InstrumentationKeyDescriptor::ConstValueMapIterator itV;

            InstrumentationDictionary *iD = sys.getInstrumentor()->getInstrumentorDictionary();

            /* Generating key/value events */
            for ( itK = iD->beginKeyMap(); itK != iD->endKeyMap(); itK++ ) {
               InstrumentationKeyDescriptor *kD = itK->second;
 
               p_file << "EVENT_TYPE" << std::endl;
               p_file << "9    " << _eventBase+kD->getId() << " " << kD->getDescription() << std::endl;
               p_file << "VALUES" << std::endl;

               // First: Ordering list of values and descriptions 
               std::map<int,std::string> lov;
               for ( itV = kD->beginValueMap(); itV != kD->endValueMap(); itV++ ) {
                  InstrumentationValueDescriptor *vD = itV->second;
                  lov.insert( make_pair( vD->getId(), vD->getDescription() ));
               }

               // Second:: Generating already ordered list of values
               std::map<int,std::string>::iterator itLoV;
               for ( itLoV = lov.begin(); itLoV != lov.end(); itLoV++ ) {
                  p_file << itLoV->first << "  " << itLoV->second << std::endl;
               }

               p_file << std::endl;
            }
            p_file << std::endl;

            /* Closing configuration file */
            p_file.close();
         }
         else std::cout << "Unable to open paraver config file" << std::endl;  
      }

      void renameFiles ()
      {
         char *new_name_prv = (char *) alloca ( 255 * sizeof (char));;
         char *new_name_pcf = (char *) alloca ( 255 * sizeof (char));;
         char *new_name_row = (char *) alloca ( 255 * sizeof (char));;

         // Check if the trace file exists
         struct stat buffer;
         int err;
         std::string trace_base = ( OS::getArg( 0 ) );
         int num = 1;
         std::string trace_suffix = "_001";
         std::string trace_extension = ".prv";
         bool trace_exists = true;

         while ( trace_exists ) {
            // Attempt to get the file attributes
            err = stat( ( trace_base + trace_suffix + trace_extension).c_str(), &buffer );
            if ( err == 0 ) {
               // The file exists
               num++;
               std::stringstream trace_num;
               trace_num << "_" << (num < 100 ? "0" : "") << (num < 10 ? "0" : "") << num;
               trace_suffix =  trace_num.str();
            } else {
               trace_exists = false;
            }
         }

         sprintf( new_name_prv, "%s%s.prv", trace_base.c_str(), trace_suffix.c_str() );
         sprintf( new_name_pcf, "%s%s.pcf", trace_base.c_str(), trace_suffix.c_str() );
         sprintf( new_name_row, "%s%s.row", trace_base.c_str(), trace_suffix.c_str() );

         /* Renaming the files */
         int result;

         result = rename( "MPITRACE_Paraver_Trace.prv"  , new_name_prv );
         if ( result != 0 ) std::cout << "Unable to rename paraver file" << std::endl;
         result = rename( "MPITRACE_Paraver_Trace.pcf"  , new_name_pcf );
         if ( result != 0 ) std::cout << "Unable to rename paraver config file" << std::endl;
         result = rename( "MPITRACE_Paraver_Trace.row"  , new_name_row );
         if ( result != 0 ) std::cout << "Unable to rename paraver row file" << std::endl;

         std::cout << "nanox: Trace MPITRACE_Paraver_Trace.prv renamed to " << trace_base << trace_suffix << ".prv." << std::endl;
      }

      void initialize ( void )
      {
         char *mpi_trace_on;

         /* check environment variable MPITRACE_ON value */
         mpi_trace_on = getenv("EXTRAE_ON");

         /* if MPITRAE_ON not defined, active it */
         if ( mpi_trace_on == NULL )
         {
            mpi_trace_on = new char[15];
            strcpy(mpi_trace_on, "EXTRAE_ON=1");
            putenv (mpi_trace_on);
         }

         /* OMPItrace initialization */
         OMPItrace_init();
      }

      void finalize ( void )
      {
         OMPItrace_fini();
         mergeParaverTraceFiles();
         createParaverConfigFile();
         renameFiles();
      }

      void addEventList ( unsigned int count, Event *events) 
      {
         struct extrae_CombinedEvents ce;

         ce.HardwareCounters = 1;
         ce.Callers = 0;
         ce.UserFunction = 0;
         ce.nEvents = 0;
         ce.nCommunications = 0;

         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            switch ( e.getType() ) {
               case STATE_START:
               case STATE_END:
               case SUBSTATE_START:
               case SUBSTATE_END:
                  ce.nEvents++;
                  break;
               case PTP_START:
               case PTP_END:
                  ce.nCommunications++;
                  // continue...
               case POINT:
               case BURST_START:
               case BURST_END:
                  ce.nEvents += e.getNumKVs();
                  break;
               default: break;
            }
         }

         ce.Types = (unsigned int *) alloca (ce.nEvents * sizeof (unsigned int));
         ce.Values = (unsigned int *) alloca (ce.nEvents * sizeof (unsigned int));
         ce.Communications = (struct extrae_UserCommunication *) alloca (ce.nCommunications * sizeof (struct extrae_UserCommunication));

         int j = 0; int k = 0;
         Event::ConstKVList kvs = NULL;

         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            unsigned int type = e.getType();
            switch ( type ) {
               case STATE_START:
                  ce.Types[j] = _eventState;
                  ce.Values[j++] = e.getState();
                  break;
               case STATE_END:
                  ce.Types[j] = _eventState;
                  ce.Values[j++] = 0;
                  break;
               case SUBSTATE_START:
                  ce.Types[j] = _eventSubState;
                  ce.Values[j++] = e.getState();
                  break;
               case SUBSTATE_END:
                  ce.Types[j] = _eventSubState;
                  ce.Values[j++] = 0;
                  break;
               case PTP_START:
               case PTP_END:
                  /* Creating PtP event */
                  if ( type == PTP_START) ce.Communications[k].type = EXTRAE_USER_SEND;
                  else ce.Communications[k].type = EXTRAE_USER_RECV;
                  ce.Communications[k].tag = e.getDomain();
                  ce.Communications[k].id = e.getId();
                  ce.Communications[k].size = e.getId(); // FIXME: just in some cases size is equal to id
                  ce.Communications[k].partner = 0;
                  k++;
                  // continue...
               case POINT:
               case BURST_START:
                  kvs = e.getKVs();
                  for ( unsigned int kv = 0 ; kv < e.getNumKVs() ; kv++,kvs++ ) {
                     ce.Types[j] = _eventBase + kvs->first;
                     ce.Values[j++] = kvs->second;
                  }
                  break;
               case BURST_END:
                  kvs = e.getKVs();
                  for ( unsigned int kv = 0 ; kv < e.getNumKVs() ; kv++,kvs++ ) {
                     ce.Types[j] = _eventBase +  kvs->first;
                     ce.Values[j++] = 0; // end
                  }
                  break;
               default: break;
            }
         }

         // if showing stacked burst is false remove duplicates
         if ( !_icLocal.showStackedBursts() ) {
            int rmValues = 0;
            for ( int i = 0; i < ce.nEvents; i++ )
            {
               for ( int j = i+1; j < ce.nEvents; j++ )
               {
                  if ( ce.Types[i] == ce.Types[j] )
                  {
                     ce.Types[i] = 0;
                     rmValues++;
                  }
               }
            }
            ce.nEvents -= rmValues;
            for ( int j = 0, i = 0; i < ce.nEvents; i++ )
            {
               while ( ce.Types[j] == 0 ) j++;
               ce.Types[i] = ce.Types[j];
               ce.Values[i] = ce.Values[j++];
            }
         }

         Extrae_emit_CombinedEvents ( &ce );
      }
#endif
};

namespace ext {

class InstrumentorParaverPlugin : public Plugin {
   public:
      InstrumentorParaverPlugin () : Plugin("Instrumentor which generates a Paraver trace.",1) {}
      ~InstrumentorParaverPlugin () {}

      virtual void config( Config &config ) {}

      void init ()
      {
         sys.setInstrumentor( new InstrumentationExtrae() );
      }
};

} // namespace ext

} // namespace nanos

nanos::ext::InstrumentorParaverPlugin NanosXPlugin;

