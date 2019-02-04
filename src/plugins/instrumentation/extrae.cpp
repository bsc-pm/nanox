/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#include "plugin.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"
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
#include <unistd.h>

#include "os.hpp"
#include "errno.h"
#include <unistd.h>

#ifdef NANOS_INSTRUMENTATION_ENABLED
#include <extrae_types.h>
#include <mpitrace_user_events.h>

#define extrae_size_t unsigned int

extern "C" {
   unsigned int nanos_ompitrace_get_max_threads ( void );
   unsigned int nanos_ompitrace_get_thread_num ( void );
   void nanos_extrae_instrumentation_barrier( void );
   unsigned int nanos_extrae_node_id();
   unsigned int nanos_extrae_num_nodes();
   void         nanos_ompitrace_instrumentation_barrier();
   void         Extrae_change_num_threads (unsigned nthreads);
}
#endif

namespace nanos {

#ifdef NANOS_INSTRUMENTATION_ENABLED
   const extrae_type_t _eventState      = 9000000;   /*!< event coding state changes */
   const extrae_type_t _eventPtPStart   = 9000001;   /*!< event coding comm start */
   const extrae_type_t _eventPtPEnd     = 9000002;   /*!< event coding comm end */
   const extrae_type_t _eventSubState   = 9000004;   /*!< event coding sub-state changes */
   const extrae_type_t _eventBase       = 9200000;   /*!< event base (used in key/value pairs) */
#endif

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
      std::string                                    _listOfTraceFileNames;
      std::string                                    _traceDirectory;        /*<< Extrae directory: EXTRAE_DIR */
      std::string                                    _traceFinalDirectory;   /*<< Extrae final directory: EXTRAE_FINAL_DIR */
      std::string                                    _traceParaverDirectory; /*<< Paraver output files directory */
      std::string                                    _traceFileName_PRV;     /*<< Paraver: file.prv */
      std::string                                    _traceFileName_PCF;     /*<< Paraver: file.pcf */
      std::string                                    _traceFileName_ROW;     /*<< Paraver: file.row */
      std::string                                    _binFileName;           /*<< Binnary file name */
      int                                            _maxThreads;
      Lock                                           _lock;
   public:
      static std::string                             _traceBaseName;
      // constructor
      InstrumentationExtrae ( ) : Instrumentation( *NEW InstrumentationContextDisabled() ) {}
      // destructor
      ~InstrumentationExtrae ( ) { }

      void secureCopy(const char *orig, std::string dest)
      {
         pid_t pid;
         int status, options = 0;

         std::cerr << "secure copy " << orig << " to " << dest << std::endl;

         pid = vfork();
         if ( pid == (pid_t) 0 ) {
            int execret = execl ( "/usr/bin/scp", "scp", orig, dest.c_str(), (char *) NULL); 
            if ( execret == -1 )
            {
               std::cerr << "Error calling /usr/bin/scp " << orig << " " << dest.c_str() << std::endl;
               exit(-1);
            }
         } else {
            if ( pid < 0 ) {
                int errsv = errno;
                message0("Error: Cannot execute mpi2prv due following error:");
                switch ( errsv ){
                   case EAGAIN:
                      message0("fork() cannot allocate sufficient memory to copy the parent's page tables and allocate a task structure for the child.");
                      break;
                   case ENOMEM:
                      message0("fork() failed to allocate the necessary kernel structures because memory is tight.");
                      break;
                   default:
                      message0("fork() unknow error.");
                      break;
                }
            } else {
               waitpid( pid, &status, options);
            }
         }
      }

      void copyFilesToMaster()
      {
         /* Removig Temporary trace files */
         char str[255];
         std::fstream p_file;

         for ( unsigned int j = 0; j < sys.getNetwork()->getNumNodes(); j++ )
         {
            if ( j == sys.getNetwork()->getNodeNum() )
            {
               p_file.open(_listOfTraceFileNames.c_str());
               //size_t found = _traceFinalDirectory.find_last_of("/");
               std::string dst = std::string(sys.getNetwork()->getMasterHostname() );

               if (p_file.is_open())
               {
                  unsigned int thread = 0;
                  while (!p_file.eof() )
                  {
                     p_file.getline (str, 255);
                     if ( strlen(str) > 0 )
                     {
                        std::string src_path( str );
                        std::size_t pos = src_path.size() ;
                        for (unsigned int i = 0; i < 2; i++ ) {
                           pos = src_path.find_last_of('/', pos - 1);
                        }
                        //int pos0 =  src_path.find_last_of('/');
                        //int pos1 =  src_path.find_first_of(' ');
                        //std::cerr << "len is " << pos1-pos0 << " pos0: " << pos0 << " total size is " << src_path.size()<< "src_path is " << src_path<< std::endl;
                        //std::string name( src_path.substr( pos0, pos1-pos0 ) );

                        for (unsigned int i = 0; i < strlen(str); i++) if ( str[i] == ' ' ) str[i] = 0x0;
                        // jbueno: cluster workaround until we get the new extrae
                        //if ( sys.getNetwork()->getNodeNum() > 0 ) {
                        //   str[ strlen(str) - 12 ] = '0';// + ( (char) ( sys.getNetwork()->getNodeNum() % 10 ) );
                        //   str[ strlen(str) - 13 ] = '0';// + ( (char) ( sys.getNetwork()->getNodeNum() / 10 ) );
                        //   str[ strlen(str) - 14 ] = '0';
                        //   str[ strlen(str) - 15 ] = '0';
                        //   str[ strlen(str) - 16 ] = '0';
                        //   str[ strlen(str) - 17 ] = '0';
                        //}
                        //std::cerr << "NAME: " << name << std::endl;
                        secureCopy(str, dst + ":" + src_path.substr( 0, pos + 1 ) /* + name */ );

                        //Copy the symbol file
                        if ( thread == 0 ) {
                           char sym_file_name[256];
                           std::size_t dot_pos = src_path.find(".mpit");
                           std::string myName = src_path.substr( src_path.find_last_of('/') + 1, dot_pos - src_path.find_last_of('/') );
                           sprintf( sym_file_name, "%s/set-0/%ssym", _traceDirectory.c_str(), myName.c_str() );
                           secureCopy(sym_file_name, dst + ":" + src_path.substr( 0, pos + 1 ) );
                        }

                        thread += 1;
                     }
                  }
                  p_file.close();
               }
               else std::cout << "Unable to open " << _listOfTraceFileNames << " file" << std::endl;  

               // copy pcf file too
               //{
               //   size_t found = _traceFinalDirectory.find_last_of("/");
               //   size_t found_pcf = _traceFileName_PCF.find_last_of("/");
               //   char number[16];
               //   sprintf(number, "%08d", sys.getNetwork()->getNodeNum() );
               //   secureCopy( _traceFileName_PCF.c_str(), dst + ":" + _traceFinalDirectory.substr(0,found+1) + number + "." + _traceFileName_PCF.substr(found_pcf+1));
               //}
            }
            nanos_extrae_instrumentation_barrier();
         }
      }

      void initialize ( void )
      {
         /* check environment variable: EXTRAE_ON */
         char *mpi_trace_on = getenv("EXTRAE_ON");
         char *mpi_trace_dir;
         char *mpi_trace_final_dir;

         /* if MPITRAE_ON not defined, active it */
         if ( mpi_trace_on == NULL ) {
            mpi_trace_on = NEW char[12];
            strcpy(mpi_trace_on, "EXTRAE_ON=1");
            putenv (mpi_trace_on);
         }

         /* check environment variable: EXTRAE_FINAL_DIR */
         mpi_trace_final_dir = getenv("EXTRAE_FINAL_DIR");
         /* if EXTRAE_FINAL_DIR not defined, active it */
         if ( mpi_trace_final_dir == NULL ) {
            mpi_trace_final_dir = NEW char[3];
            strcpy(mpi_trace_final_dir, "./");
         }

         /* check environment variable: EXTRAE_DIR */
         mpi_trace_dir = getenv("EXTRAE_DIR");
         /* if EXTRAE_DIR not defined, active it */
         if ( mpi_trace_dir == NULL ) {
            mpi_trace_dir = NEW char[3];
            strcpy(mpi_trace_dir, "./");
         }

         _traceDirectory = mpi_trace_dir;
         _traceFinalDirectory = mpi_trace_final_dir;
         _listOfTraceFileNames = _traceFinalDirectory + "/TRACE.mpits";


        // Common thread information
        Extrae_set_threadid_function ( nanos_ompitrace_get_thread_num );
        Extrae_set_numthreads_function ( nanos_ompitrace_get_max_threads );

        // Cluster specific information
        if ( sys.usingCluster() ) {
           Extrae_set_taskid_function ( nanos_extrae_node_id );
           Extrae_set_numtasks_function ( nanos_extrae_num_nodes );
           Extrae_set_barrier_tasks_function ( nanos_ompitrace_instrumentation_barrier );
        }
#ifdef HAVE_MPI_H
        char *offload_trace_on = getenv("NX_OFFLOAD_INSTRUMENTATION");
        if (offload_trace_on != NULL){ 
           //MPI plugin init will initialize extrae...
           sys.loadPlugin("arch-mpi");
        } else {
#endif
            /* Regular SMP OMPItrace initialization */      
            OMPItrace_init();      
#ifdef HAVE_MPI_H
        }
#endif
        
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
        _maxThreads = sys.getSMPPlugin()->getNumThreads();
        Extrae_change_num_threads( _maxThreads );
      }
      void doLs(std::string dest)
      {
         pid_t pid;
         int status, options = 0;

         std::cerr << "list directory " << dest << std::endl;

         pid = vfork();
         if ( pid == (pid_t) 0 ) {
            dup2(2, 1);
            int execret = execl ( "/bin/ls", "ls", dest.c_str(), (char *) NULL); 
            if ( execret == -1 )
            {
               std::cerr << "Error calling /bin/ls " << dest.c_str() << std::endl;
               exit(-1);
            }
         } else {
            if ( pid < 0 ) {
                int errsv = errno;
                message0("Error: Cannot execute mpi2prv due following error:");
                switch ( errsv ){
                   case EAGAIN:
                      message0("fork() cannot allocate sufficient memory to copy the parent's page tables and allocate a task structure for the child.");
                      break;
                   case ENOMEM:
                      message0("fork() failed to allocate the necessary kernel structures because memory is tight.");
                      break;
                   default:
                      message0("fork() unknow error.");
                      break;
                }
            } else {
               waitpid( pid, &status, options);
            }
         }
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
            if ( kD->getId() == 0 ) continue;
            extrae_type_t type = _eventBase+kD->getId(); 
            char *type_desc = ( char *) alloca(sizeof(char) * (kD->getDescription().size() + 1) );
            strncpy ( type_desc, kD->getDescription().c_str(), kD->getDescription().size()+1 );
            unsigned nval = kD->getSize();
            if ( kD->getId() == usr_functLocation ) {
               for ( itV = kD->beginValueMap(); itV != kD->endValueMap(); itV++ ) {
                  // Parsing event description
                  std::string description = iD->getValueDescription( kD->getId(), (itV->second)->getId() );
                  size_t pos1 = description.find_first_of("@");
                  size_t pos2 = description.find_first_of("@",pos1+1);
                  size_t pos3 = description.find_first_of("@",pos2+1);
                  pos3 = (pos3 == std::string::npos ? description.length() : pos3-1);
                  int  line = atoi ( (description.substr(pos2+1, pos3)).c_str());
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
               Extrae_define_event_type( (extrae_type_t *) &type, type_desc, &val_id, values, val_desc);

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
               "DATA TRANSFER ISSUE", "CACHE ALLOC/FREE", "YIELD", "ACQUIRING LOCK", "CONTEXT SWITCH",
               "FILL COLOR", "WAKING UP", "STOPPED", "SYNCED RUNNING", "DEBUG"};

            for ( i = 0; i < (nval - 1); i++ ) { // Do not show the DEBUG state
               values[i] = i;
               val_desc[i] = (char *) nanos_event_state_value_str[i].c_str();
            }
            values[i] = 27;
            val_desc[i++] = (char *) "EXTRAE I/O";

            unsigned extrae_zero = 0;
            Extrae_define_event_type( (extrae_type_t *) &_eventState, (char *) "Thread state: ", &nval, values, val_desc );
            Extrae_define_event_type( (extrae_type_t *) &_eventPtPStart, (char *) "Point-to-point origin", &extrae_zero, NULL, NULL );
            Extrae_define_event_type( (extrae_type_t *) &_eventPtPEnd, (char *) "Point-to-point destination", &extrae_zero, NULL, NULL );
            Extrae_define_event_type( (extrae_type_t *) &_eventSubState, (char *) "Thread sub-state", &nval, values, val_desc );
         }

         //If offloading, MPI will finish the trace
         char *offload_trace_on = getenv("NX_OFFLOAD_INSTRUMENTATION");
         if (offload_trace_on == NULL){ 
            OMPItrace_fini();
         }

         if ( sys.usingCluster() ) {
            copyFilesToMaster();
         }
      }

      void disable( void ) { Extrae_shutdown(); }
      void enable( void ) { Extrae_restart(); }

      void addEventList ( unsigned int count, Event *events) 
      {
         extrae_combined_events_t ce;
         InstrumentationDictionary *iD = sys.getInstrumentation()->getInstrumentationDictionary();
         bool _stateEnabled = sys.getInstrumentation()->isStateEnabled();
         bool _ptpEnabled = sys.getInstrumentation()->isPtPEnabled();

         ce.HardwareCounters = 0;
         ce.Callers = 0;
         ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
         ce.nEvents = 0;
         ce.nCommunications = 0;
  
         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            nanos_event_type_t type = e.getType();
            nanos_event_key_t key = e.getKey();
            switch ( type ) {
               case NANOS_STATE_START:
               case NANOS_STATE_END:
               case NANOS_SUBSTATE_START:
               case NANOS_SUBSTATE_END:
                  if ( !_stateEnabled ) continue;
                  ce.nEvents++;
                  break;
               case NANOS_PTP_START:
               case NANOS_PTP_END:
                  if ( !_ptpEnabled ) continue;
                  ce.nCommunications++;
                  break;
               case NANOS_POINT:
               case NANOS_BURST_START:
               case NANOS_BURST_END:
                  if ( key == 0 ) continue;
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
                  if ( !_stateEnabled ) continue;
                  ce.Types[j] = _eventState;
                  ce.Values[j++] = e.getState();
                  break;
               case NANOS_STATE_END:
                  if ( !_stateEnabled ) continue;
                  ce.Types[j] = _eventState;
                  ce.Values[j++] = 0;
                  break;
               case NANOS_SUBSTATE_START:
                  if ( !_stateEnabled ) continue;
                  ce.Types[j] = _eventSubState;
                  ce.Values[j++] = e.getState();
                  break;
               case NANOS_SUBSTATE_END:
                  if ( !_stateEnabled ) continue;
                  ce.Types[j] = _eventSubState;
                  ce.Values[j++] = 0;
                  break;
               case NANOS_PTP_START:
               case NANOS_PTP_END:
                  if ( !_ptpEnabled ) continue;
                  /* Creating PtP event */
                  ckey = e.getKey();

                  if ( type == NANOS_PTP_START) ce.Communications[k].type = EXTRAE_USER_SEND;
                  else ce.Communications[k].type = EXTRAE_USER_RECV;
                  ce.Communications[k].tag = e.getDomain();
                  ce.Communications[k].id = e.getId();

                  if ( ckey != 0 && ckey == sizeKey ) ce.Communications[k].size = e.getValue();
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
                  // Add hwc only for user-funct events
                  if ( ckey != 0 && ( ckey ==  getInstrumentationDictionary()->getEventKey("user-funct-location") || ckey ==  getInstrumentationDictionary()->getEventKey("cpuid") ) )
                     ce.HardwareCounters = 1;
                  break;
               case NANOS_BURST_END:
                  ckey = e.getKey();
                  if (  ckey != 0 ) { 
                     ce.Types[j] = _eventBase + ckey;
                     ce.Values[j++] = 0; // end
                  }
                  if ( ckey !=0 && (ckey ==  getInstrumentationDictionary()->getEventKey("user-funct-location") || ckey ==  getInstrumentationDictionary()->getEventKey("cpuid") ) )
                     ce.HardwareCounters = 1;
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

         if ( ce.nEvents == 0 && ce.nCommunications == 0 ) return;

//FIXME: to remove when closing #1034
#if 0
         fprintf(stderr,"\nEvents: ");
         for ( extrae_size_t jj = 0; jj < ce.nEvents; jj++ )
           fprintf(stderr,"%d, ", (int)ce.Types[jj]);
         fprintf(stderr,"\n");

         fprintf(stderr,"\nCommunications: ");
         for ( extrae_size_t jj = 0; jj < ce.nCommunications; jj++ )
           fprintf(stderr,"%d, ", (int) ce.Communications[jj].type);
         fprintf(stderr,"\n");
#endif

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
         // Extrae_change_num_threads involves memory allocation and file creation,
         // thus the function call must be lock-protected
         _lock.acquire();
         Extrae_change_num_threads( ++_maxThreads );
         _lock.release();
      }

#endif
};

#ifdef NANOS_INSTRUMENTATION_ENABLED
std::string InstrumentationExtrae::_traceBaseName = std::string("");
#endif

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
