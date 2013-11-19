#include "plugin.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"
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
#include "errno.h"
#include <unistd.h>

#ifndef EXTRAE_VERSION
#warning Extrae library version is not supported (use >= 2.3):
#else
#  define NANOX_EXTRAE_SUPPORTED_VERSION
#  if EXTRAE_VERSION_MAJOR(EXTRAE_VERSION) == 2 /************* version 2.x.x */
#      define extrae_size_t unsigned int

#    if EXTRAE_VERSION_MINOR(EXTRAE_VERSION) == 2 /*********** version 2.2.x */ 
#      warning Extrae library version is not supported (use >= 2.3):
#      undef NANOX_EXTRAE_SUPPORTED_VERSION
#    endif /*------------------------------------------------- version 2.2.x */

#    if EXTRAE_VERSION_MINOR(EXTRAE_VERSION) == 3 /*********** version 2.3.x */
#      if EXTRAE_VERSION_REVISION(EXTRAE_VERSION) == 0 /****** version 2.3.0 */
#         define NANOX_EXTRAE_OLD_DEFINE_TYPE
#      endif /*----------------------------------------------- version 2.3.0 */
#      if EXTRAE_VERSION_REVISION(EXTRAE_VERSION) == 1 /****** version 2.3.1 */
#         define NANOX_EXTRAE_OLD_DEFINE_TYPE
#      endif /*----------------------------------------------- version 2.3.1 */
#      if EXTRAE_VERSION_REVISION(EXTRAE_VERSION) == 2 /****** version 2.3.2 */
#         define NANOX_EXTRAE_OLD_DEFINE_TYPE
#      endif /*----------------------------------------------- version 2.3.2 */
#      if EXTRAE_VERSION_REVISION(EXTRAE_VERSION) == 3 /****** version 2.3.3 */
#         define NANOX_EXTRAE_OLD_DEFINE_TYPE
#      endif /*----------------------------------------------- version 2.3.3 */
#      if EXTRAE_VERSION_REVISION(EXTRAE_VERSION) == 4 /****** version 2.3.4 */
#      endif /*----------------------------------------------- version 2.3.4 */
#    endif /*------------------------------------------------- version 2.3.x */

#  endif /*--------------------------------------------------- version 2.x.x */
#  if EXTRAE_VERSION_MAJOR(EXTRAE_VERSION) == 3 /************* version 3.x.x */
#      define extrae_size_t unsigned int
#  endif /*--------------------------------------------------- version 3.x.x */
#endif

#ifdef NANOX_EXTRAE_SUPPORTED_VERSION
extern "C" {
   unsigned int nanos_ompitrace_get_max_threads ( void );
   unsigned int nanos_ompitrace_get_thread_num ( void );
   unsigned int nanos_extrae_node_id();
   unsigned int nanos_extrae_num_nodes();
   void         nanos_ompitrace_instrumentation_barrier();
   void         Extrae_change_num_threads (unsigned nthreads);
}

namespace nanos {

   const extrae_type_t _eventState      = 9000000;   /*!< event coding state changes */
   const extrae_type_t _eventPtPStart   = 9000001;   /*!< event coding comm start */
   const extrae_type_t _eventPtPEnd     = 9000002;   /*!< event coding comm end */
   const extrae_type_t _eventSubState   = 9000004;   /*!< event coding sub-state changes */
   const extrae_type_t _eventBase       = 9200000;   /*!< event base (used in key/value pairs) */

class InstrumentationExtrae: public Instrumentation 
{
#ifndef NANOS_INSTRUMENTATION_ENABLED
   public:
      // constructor
      InstrumentationExtrae( ) : Instrumentation() {}

      // destructor
      ~InstrumentationExtrae() {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void ) {}
      void finalize( void ) {}
      void disable( void ) {}
      void enable( void ) {}
      void addEventList ( unsigned int count, Event *events ) {}
      void addResumeTask( WorkDescriptor &w ) {}
      void addSuspendTask( WorkDescriptor &w ) {}
      void threadStart( BaseThread &thread ) {}
      void threadFinish ( BaseThread &thread ) {}
      void incrementMaxThreads( void ) {}
#else
   private:
      int                                            _maxThreads;
   public:
      // constructor
      InstrumentationExtrae ( ) : Instrumentation( *NEW InstrumentationContextDisabled() ) {}
      // destructor
      ~InstrumentationExtrae ( ) { }

      void modifyParaverRowFile()
      {
#if 0
         // rename ROW file to a temporary file
         std::string line;
         std::string _traceFileName_ROW_tmp = _traceFileName_ROW + "__tmp";
         rename ( _traceFileName_ROW.c_str(), _traceFileName_ROW_tmp.c_str() );

         // Input file: temporary file
         std::ifstream i_file;
         i_file.open ( _traceFileName_ROW_tmp.c_str(), std::ios::in );

         // Output file: paraver config 
         std::ofstream o_file;
         o_file.open ( _traceFileName_ROW.c_str(), std::ios::out | std::ios::app);

         if ( o_file.is_open() && i_file.is_open() ) {
            bool cont = true;
            bool print = true;
            while ( cont ) {
               cont = getline ( i_file, line );
               if ( print == true ) {
                  // printing was alredy enabled, so disable if...
                  print = print && line.find("LEVEL THREAD"); // ... found LEVEL THREAD section
                  print = print && line.find("LEVEL CPU"); // ... found LEVEL CPU section
               } else {
                  // printing was already disabled so enabled if...
                  print = !line.find("LEVEL NODE"); // ... found LEVEL NODE section
               }

               if ( print ) o_file << line << std::endl;
            }

            // Adding thread info
            unsigned int num_threads = sys.getNumWorkers();
            o_file << "LEVEL CPU SIZE " << num_threads << std::endl;
            for ( unsigned int i = 0; i < num_threads; i++ ) {
               o_file << sys.getWorker(i)->getDescription() << std::endl;
            }
            o_file << std::endl;

            o_file.close();
            i_file.close();

            remove ( _traceFileName_ROW_tmp.c_str() );
         } else {
            if (o_file.is_open()) o_file.close();
            if (i_file.is_open()) i_file.close();
            message0("Unable to open paraver config file");  
            rename ( _traceFileName_ROW_tmp.c_str(), _traceFileName_ROW.c_str() );
         }
#endif
      }

      void initialize ( void )
      {
         /* check environment variable: EXTRAE_ON */
         char *mpi_trace_on = getenv("EXTRAE_ON");
         /* if MPITRAE_ON not defined, active it */
         if ( mpi_trace_on == NULL ) {
            mpi_trace_on = NEW char[12];
            strcpy(mpi_trace_on, "EXTRAE_ON=1");
            putenv (mpi_trace_on);
         }

        // Common thread information
        Extrae_set_threadid_function ( nanos_ompitrace_get_thread_num );
        Extrae_set_numthreads_function ( nanos_ompitrace_get_max_threads );

        // Cluster specific information
        Extrae_set_taskid_function ( nanos_extrae_node_id );
        Extrae_set_numtasks_function ( nanos_extrae_num_nodes );
        Extrae_set_barrier_tasks_function ( nanos_ompitrace_instrumentation_barrier );

        /* OMPItrace initialization */
        OMPItrace_init();

        Extrae_register_codelocation_type( 9200011, 9200021, "User Function Name", "User Function Location" );

        Extrae_register_stacked_type( (extrae_type_t) _eventState );
        InstrumentationDictionary::ConstKeyMapIterator itK;
        InstrumentationDictionary *iD = sys.getInstrumentation()->getInstrumentationDictionary();

        /* Generating key/value events */
        for ( itK = iD->beginKeyMap(); itK != iD->endKeyMap(); itK++ ) {
           InstrumentationKeyDescriptor *kD = itK->second;
           if (kD->isStacked()) {
              Extrae_register_stacked_type( (extrae_type_t) _eventBase+kD->getId() );
           }
        }

        /* Keep current number of threads */
        _maxThreads = sys.getNumThreads();
      }

      void finalize ( void )
      {
         /* Getting Instrumentation Dictionary */
         InstrumentationDictionary::ConstKeyMapIterator itK;
         InstrumentationKeyDescriptor::ConstValueMapIterator itV;
         InstrumentationDictionary *iD = sys.getInstrumentation()->getInstrumentationDictionary();
	      nanos_event_key_t usr_functName = iD->getEventKey("user-funct-name");
	      nanos_event_key_t usr_functLocation = iD->getEventKey("user-funct-location");

         for ( itK = iD->beginKeyMap(); itK != iD->endKeyMap(); itK++ ) {
            InstrumentationKeyDescriptor *kD = itK->second;
            extrae_type_t type = _eventBase+kD->getId(); 
            char *type_desc = ( char *) alloca(sizeof(char) * (kD->getDescription().size() + 1) );
            strncpy ( type_desc, kD->getDescription().c_str(), kD->getDescription().size()+1 );
            unsigned nval = kD->getSize();
            if ( kD->getId() == usr_functLocation ) {
               for ( itV = kD->beginValueMap(); itV != kD->endValueMap(); itV++ ) {
                  // Parsing event description
                  std::string description = iD->getValueDescription( kD->getId(), (itV->second)->getId() );
                  int pos1 = description.find_first_of("@");
                  int pos2 = description.find_first_of("@",pos1+1);
                  int length = description.size();
                  int  line = atoi ( (description.substr(pos2+1, length)).c_str());
                  Extrae_register_function_address ( 
                     (void *) (itV->second)->getId(),
                     (char *) description.substr(0,pos1).c_str(),
                     (char *) description.substr(pos1+1,(pos2-pos1-1)).c_str(),
                     (unsigned) line
                  );
               }
            } else if (kD->getId() == usr_functName ) {
               // DO Nothing
            } else {
               extrae_value_t *values = (extrae_value_t *) alloca(sizeof(extrae_value_t) * nval);
               char **val_desc = (char **) alloca(sizeof(char *) * nval);
               unsigned val_id = 0;
               for ( itV = kD->beginValueMap(); itV != kD->endValueMap(); itV++ ) {
                  InstrumentationValueDescriptor *vD = itV->second;
                  values[val_id] = vD->getId();
                  val_desc[val_id] = (char *) alloca(sizeof(char) * (vD->getDescription().size() + 1) );
                  strncpy(val_desc[val_id], vD->getDescription().c_str(), vD->getDescription().size()+1 );
                  val_id++;
               }
#ifdef NANOX_EXTRAE_OLD_DEFINE_TYPE
               Extrae_define_event_type( (extrae_type_t) type, type_desc, val_id, values, val_desc);
#else
               Extrae_define_event_type( (extrae_type_t *) &type, type_desc, &val_id, values, val_desc);
#endif

            }
         }
         /* HARDCODED values */
         {
            unsigned nval = NANOS_EVENT_STATE_TYPES;
            extrae_value_t *values = (extrae_value_t *) alloca( sizeof(extrae_value_t) * nval );
            char **val_desc = (char **) alloca( sizeof(char *) * nval );
            unsigned int i = 0;
            static std::string nanos_event_state_value_str[] = {"NOT CREATED", "NOT RUNNING", 
               "STARTUP", "SHUTDOWN", "ERROR", "IDLE",
               "RUNTIME", "RUNNING", "SYNCHRONIZATION", "SCHEDULING", "CREATION",
               "DATA TRANSFER TO DEVICE", "DATA TRANSFER TO HOST", "LOCAL DATA TRANSFER IN DEVICE",
               "DATA TRANSFER TO DEVICE", "DATA TRANSFER TO HOST", "LOCAL DATA TRANSFER IN DEVICE",
               "CACHE ALLOC/FREE", "YIELD", "ACQUIRING LOCK", "CONTEXT SWITCH", "DEBUG"};

            for ( i = 0; i < (nval - 1); i++ ) { // Do not show the DEBUG state
               values[i] = i;
               val_desc[i] = (char *) nanos_event_state_value_str[i].c_str();
            }
            values[i] = 27;
            val_desc[i++] = (char *) "EXTRAE I/O";

#ifdef NANOX_EXTRAE_OLD_DEFINE_TYPE
            Extrae_define_event_type( (extrae_type_t ) _eventState, (char *) "Thread state: ", nval, values, val_desc );
            Extrae_define_event_type( (extrae_type_t ) _eventPtPStart, (char *) "Point-to-point origin", 0, NULL, NULL );
            Extrae_define_event_type( (extrae_type_t ) _eventPtPEnd, (char *) "Point-to-point destination", 0, NULL, NULL );
            Extrae_define_event_type( (extrae_type_t ) _eventSubState, (char *) "Thread sub-state", nval, values, val_desc );
#else
            unsigned extrae_zero = 0;
            Extrae_define_event_type( (extrae_type_t *) &_eventState, (char *) "Thread state: ", &nval, values, val_desc );
            Extrae_define_event_type( (extrae_type_t *) &_eventPtPStart, (char *) "Point-to-point origin", &extrae_zero, NULL, NULL );
            Extrae_define_event_type( (extrae_type_t *) &_eventPtPEnd, (char *) "Point-to-point destination", &extrae_zero, NULL, NULL );
            Extrae_define_event_type( (extrae_type_t *) &_eventSubState, (char *) "Thread sub-state", &nval, values, val_desc );
#endif
         }

         OMPItrace_fini();
         modifyParaverRowFile();
      }

      void disable( void ) { Extrae_shutdown(); }
      void enable( void ) { Extrae_restart(); }

      void addEventList ( unsigned int count, Event *events) 
      {
         extrae_combined_events_t ce;
         InstrumentationDictionary *iD = sys.getInstrumentation()->getInstrumentationDictionary();

         ce.HardwareCounters = 1;
         ce.Callers = 0;
         ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
         ce.nEvents = 0;
         ce.nCommunications = 0;
  
         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            switch ( e.getType() ) {
               case NANOS_STATE_START:
               case NANOS_STATE_END:
               case NANOS_SUBSTATE_START:
               case NANOS_SUBSTATE_END:
                  ce.nEvents++;
                  break;
               case NANOS_PTP_START:
               case NANOS_PTP_END:
                  ce.nCommunications++;
                  break;
               case NANOS_POINT:
               case NANOS_BURST_START:
               case NANOS_BURST_END:
                  ce.nEvents++;
                  break;
               default: break;
            }
         }

         ce.Types = (extrae_type_t *) alloca (ce.nEvents * sizeof (extrae_type_t));
         ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
         ce.Communications = (extrae_user_communication_t *) alloca (ce.nCommunications * sizeof ( extrae_user_communication_t));

         int j = 0; int k = 0;
         nanos_event_key_t ckey = 0;
         extrae_value_t cvalue = 0;
         nanos_event_key_t sizeKey = iD->getEventKey("xfer-size");

         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            unsigned int type = e.getType();
            switch ( type ) {
               case NANOS_STATE_START:
                  ce.Types[j] = _eventState;
                  ce.Values[j++] = e.getState();
                  break;
               case NANOS_STATE_END:
                  ce.Types[j] = _eventState;
                  ce.Values[j++] = 0;
                  break;
               case NANOS_SUBSTATE_START:
                  ce.Types[j] = _eventSubState;
                  ce.Values[j++] = e.getState();
                  break;
               case NANOS_SUBSTATE_END:
                  ce.Types[j] = _eventSubState;
                  ce.Values[j++] = 0;
                  break;
               case NANOS_PTP_START:
               case NANOS_PTP_END:
                  /* Creating PtP event */
                  if ( type == NANOS_PTP_START) ce.Communications[k].type = EXTRAE_USER_SEND;
                  else ce.Communications[k].type = EXTRAE_USER_RECV;
                  ce.Communications[k].tag = e.getDomain();
                  ce.Communications[k].id = e.getId();

                  ckey = e.getKey();
                  if ( ckey == sizeKey ) ce.Communications[k].size = e.getValue();
                  else ce.Communications[k].size = e.getId();

                  if ( e.getPartner() == NANOX_INSTRUMENTATION_PARTNER_MYSELF ) {
                     ce.Communications[k].partner = EXTRAE_COMM_PARTNER_MYSELF;
                  } else {
                     ce.Communications[k].partner = (extrae_comm_partner_t) e.getPartner();
                  }

                  k++;
                  break;
               case NANOS_POINT:
               case NANOS_BURST_START:
                  ckey = e.getKey();
                  cvalue = e.getValue();
                  if (  ckey != 0 ) { 
                     ce.Types[j] = _eventBase + ckey;
                     ce.Values[j++] = cvalue;
                  }
                  break;
               case NANOS_BURST_END:
                  ckey = e.getKey();
                  if (  ckey != 0 ) { 
                     ce.Types[j] = _eventBase + ckey;
                     ce.Values[j++] = 0; // end
                  }
                  break;
               default: break;
            }
         }

         // if showing stacked burst is false remove duplicates
         if ( !_instrumentationContext.showStackedBursts() ) {
            int rmValues = 0;
            for ( extrae_size_t i = 0; i < ce.nEvents; i++ )
            {
               for ( extrae_size_t jj = i+1; jj < ce.nEvents; jj++ )
               {
                  if ( ce.Types[i] == ce.Types[jj] )
                  {
                     ce.Types[i] = 0;
                     rmValues++;
                  }
               }
            }
            ce.nEvents -= rmValues;
            for ( extrae_size_t jj = 0, i = 0; i < ce.nEvents; i++ )
            {
               while ( ce.Types[jj] == 0 ) jj++;
               ce.Types[i] = ce.Types[jj];
               ce.Values[i] = ce.Values[jj++];
            }
         }

         Extrae_emit_CombinedEvents ( &ce );
      }
      void addResumeTask( WorkDescriptor &w )
      {
          Extrae_resume_virtual_thread ( w.getId() );
      }

      void addSuspendTask( WorkDescriptor &w, bool last )
      {
         Extrae_suspend_virtual_thread ();
      }

      void threadStart( BaseThread &thread ) {}
      void threadFinish ( BaseThread &thread ) {}

      void incrementMaxThreads( void )
      {
         Extrae_change_num_threads( ++_maxThreads );
      }

#endif
};

namespace ext {

class InstrumentationParaverPlugin : public Plugin {
   public:
      InstrumentationParaverPlugin () : Plugin("Instrumentation which generates a Paraver trace.",1) {}
      ~InstrumentationParaverPlugin () {}

      void config( Config &cfg ) {}

      void init ()
      {
         sys.setInstrumentation( NEW InstrumentationExtrae() );
      }
};

} // namespace ext

} // namespace nanos

DECLARE_PLUGIN("instrumentation-paraver",nanos::ext::InstrumentationParaverPlugin);

#endif
