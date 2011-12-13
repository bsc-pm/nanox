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

/* NANOX_EXTRAE_DEFINE_CALLBACKS will allow to define which are our external services
 * to specify the execution environment ( thread id, total threads, etc ). They are
 * defined on Extrae 2.2.1 and above */
#define NANOX_EXTRAE_DEFINE_CALLBACKS

#ifndef EXTRAE_VERSION
#warning Extrae library version is not supported (use >= 2.2.0):
#else
#  if EXTRAE_VERSION_MAJOR(EXTRAE_VERSION) == 2 /* version 2.x.x */
#      define NANOX_EXTRAE_DISCARD_SUSPEND         //FIXME: Using old instrumentation fashion
#      define NANOX_EXTRAE_DISCARD_RESUME          //FIXME: Using old instrumentation fashion
#      define NANOX_EXTRAE_STACKED_CONTEXT_SWITCH  //FIXME: Using old instrumentation fashion
#      define extrae_size_t unsigned int
#      define NANOX_EXTRAE_DISCARD_THREAD_NAME
#    if EXTRAE_VERSION_MINOR(EXTRAE_VERSION) == 2 /* version 2.2.x */
#      if EXTRAE_VERSION_REVISION(EXTRAE_VERSION) == 0 /* version 2.2.0 */
#      define NANOX_EXTRAE_SUPPORTED_VERSION
#      undef  NANOX_EXTRAE_DEFINE_CALLBACKS
#      endif
#    endif
#  endif
#endif

#ifdef NANOX_EXTRAE_SUPPORTED_VERSION
extern "C" {
   unsigned int nanos_ompitrace_get_max_threads ( void );
   unsigned int nanos_ompitrace_get_thread_num ( void );
   unsigned int nanos_extrae_node_id();
   unsigned int nanos_extrae_num_nodes();
   void         nanos_ompitrace_instrumentation_barrier();
}

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
   public: /* must be updated by Configure */
      static std::string                             _traceBaseName;
      static std::string                             _postProcessScriptPath;
      static bool                                    _keepMpits; /*<< Keeps mpits temporary files (default = no)*/
      static bool                                    _skipMerge; /*<< Skip merge phase and keeps mpits temporary files (default = no)*/
   public:
      // constructor
#ifdef NANOX_EXTRAE_STACKED_CONTEXT_SWITCH
      InstrumentationExtrae ( ) : Instrumentation( *NEW InstrumentationContextStackedStatesAndBursts() ) {}
#else
      InstrumentationExtrae ( ) : Instrumentation( *NEW InstrumentationContextDisabled() ) {}
#endif
      // destructor
      ~InstrumentationExtrae ( ) { }

      int recursiveMkdir( const char *dir, mode_t mode)
      {
         int i=0, err=0;
         char tmp[256];

         while ( dir[i]!=0 && dir[i]=='/'){tmp[i] = dir[i];i++;}

         while ( dir[i]!=0 ) {
            while ( dir[i]!=0 && dir[i]!='/'){tmp[i] = dir[i];i++;}
            tmp[i] = 0;
            err = mkdir(tmp, mode);
            /* do not throwing intermediate dir's creation errors */
            /* if (err!= 0) return err; */
            while ( dir[i]!=0 && dir[i]=='/'){tmp[i] = dir[i];i++;}
         }
         return err;
      }

      void mergeParaverTraceFiles ()
      {
         char str[255];
         int status, options = 0;
         pid_t pid;

         // Merging trace files
         strcpy(str, MPITRACE_BIN);
         strcat(str, "/mpi2prv");

         pid = fork();
         if ( pid == (pid_t) 0 ) {
            int result = execl ( str, "mpi2prv", "-f", _listOfTraceFileNames.c_str(), "-o", _traceFileName_PRV.c_str(), "-e", _binFileName.c_str(), (char *) NULL); 
            exit(result);
         }
         else {
            waitpid( pid, &status, options);
            if ( status != 0 ) message0("Error while merging trace (mpi2prv returns: " << status << ")");
         }
      }

      void postProcessTraceFile ()
      {
         char str[255];
         int status, options = 0;
         pid_t pid;

         if ( _postProcessScriptPath == "" ) {
            strcpy(str, PREFIX);
            strcat(str,"/bin/extrae_post_process.sh");
         } else {
            strcpy(str, _postProcessScriptPath.c_str());
            strcat(str,"/extrae_post_process.sh");
         }

         pid = fork();
         if ( pid == (pid_t) 0 ) {
            int result = execlp ( "sh", "sh", str, _traceFileName_PRV.c_str(), (char *) NULL); 
            exit(result);
         }
         else waitpid( pid, &status, options);

         if ( status != 0 ) {
            message0("Error in trace post-process. Trace generated but might be incorrect");
         }
      }

      void modifyParaverConfigFile()
      {
         // Writing paraver config 
         std::fstream p_file;
         p_file.open ( _traceFileName_PCF.c_str(), std::ios::out | std::ios::app);
         if (p_file.is_open())
         {
            /* Event: State */
            p_file << "EVENT_TYPE" << std::endl;
            p_file << "9    " << _eventState  << "    Thread state: " << std::endl;
            p_file << "VALUES" << std::endl;
            p_file << NANOS_NOT_CREATED      << "     NOT CREATED" << std::endl;
            p_file << NANOS_NOT_RUNNING      << "     NOT RUNNING" << std::endl;
            p_file << NANOS_STARTUP          << "     STARTUP" << std::endl;
            p_file << NANOS_SHUTDOWN         << "     SHUTDOWN" << std::endl;
            p_file << NANOS_ERROR            << "     ERROR" << std::endl;
            p_file << NANOS_IDLE             << "     IDLE" << std::endl;
            p_file << NANOS_RUNTIME          << "     RUNTIME" << std::endl;
            p_file << NANOS_RUNNING          << "     RUNNING" << std::endl;
            p_file << NANOS_SYNCHRONIZATION  << "     SYNCHRONIZATION" << std::endl;
            p_file << NANOS_SCHEDULING       << "     SCHEDULING" << std::endl;
            p_file << NANOS_CREATION         << "     CREATION" << std::endl;
            p_file << NANOS_MEM_TRANSFER_IN  << "     DATA TRANSFER TO DEVICE" << std::endl;
            p_file << NANOS_MEM_TRANSFER_OUT << "     DATA TRANSFER TO HOST" << std::endl;
            p_file << NANOS_MEM_TRANSFER_LOCAL << "     LOCAL DATA TRANSFER IN DEVICE" << std::endl;
            p_file << NANOS_MEM_TRANSFER_DEVICE_IN << "     DATA TRANSFER TO DEVICE" << std::endl;
            p_file << NANOS_MEM_TRANSFER_DEVICE_OUT << "     DATA TRANSFER TO HOST" << std::endl;
            p_file << NANOS_MEM_TRANSFER_DEVICE_LOCAL << "     LOCAL DATA TRANSFER IN DEVICE" << std::endl;
            p_file << NANOS_CACHE            << "     CACHE ALLOC/FREE" << std::endl;
            p_file << NANOS_YIELD            << "     YIELD" << std::endl;
            p_file << NANOS_ACQUIRING_LOCK   << "     ACQUIRING LOCK" << std::endl;
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
            p_file << NANOS_NOT_CREATED      << "     NOT CREATED" << std::endl;
            p_file << NANOS_NOT_RUNNING      << "     NOT RUNNING" << std::endl;
            p_file << NANOS_STARTUP          << "     STARTUP" << std::endl;
            p_file << NANOS_SHUTDOWN         << "     SHUTDOWN" << std::endl;
            p_file << NANOS_ERROR            << "     ERROR" << std::endl;
            p_file << NANOS_IDLE             << "     IDLE" << std::endl;
            p_file << NANOS_RUNTIME          << "     RUNTIME" << std::endl;
            p_file << NANOS_RUNNING          << "     RUNNING" << std::endl;
            p_file << NANOS_SYNCHRONIZATION  << "     SYNCHRONIZATION" << std::endl;
            p_file << NANOS_SCHEDULING       << "     SCHEDULING" << std::endl;
            p_file << NANOS_CREATION         << "     CREATION" << std::endl;
            p_file << NANOS_MEM_TRANSFER_IN  << "     DATA TRANSFER TO DEVICE" << std::endl;
            p_file << NANOS_MEM_TRANSFER_OUT << "     DATA TRANSFER TO HOST" << std::endl;
            p_file << NANOS_MEM_TRANSFER_LOCAL << "     LOCAL DATA TRANSFER IN DEVICE" << std::endl;
            p_file << NANOS_CACHE            << "     CACHE ALLOC/FREE" << std::endl;
            p_file << NANOS_YIELD            << "     YIELD" << std::endl;
            p_file << NANOS_ACQUIRING_LOCK   << "     ACQUIRING LOCK" << std::endl;
            p_file << std::endl;

            /* Getting Instrumentation Dictionary */
            InstrumentationDictionary::ConstKeyMapIterator itK;
            InstrumentationKeyDescriptor::ConstValueMapIterator itV;

            InstrumentationDictionary *iD = sys.getInstrumentation()->getInstrumentationDictionary();

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
         else message0("Unable to open paraver config file");
      }

      void modifyParaverRowFile()
      {
#ifndef NANOX_EXTRAE_DISCARD_THREAD_NAME
         unsigned int num_threads = sys.getNumWorkers();
         // Writing paraver config 
         std::fstream p_file;
         p_file.open ( _traceFileName_ROW.c_str(), std::ios::out | std::ios::app);
         if (p_file.is_open())
         {
            /* Adding thread info */
            p_file << std::endl;
            p_file << "LEVEL THREAD SIZE " << num_threads << std::endl;
            for ( unsigned int i = 0; i < num_threads; i++ ) {
               p_file << sys.getWorker(i)->getDescription() << std::endl;
            }
            p_file << std::endl;

            /* Closing configuration file */
            p_file.close();
         }
         else message0("Unable to open paraver config file");  
#endif
      }

      void removeTemporaryFiles()
      {
         char *file_name    = (char *) alloca ( 255 * sizeof (char));
         bool file_exists; int num;

         if ( !_keepMpits && !_skipMerge ) {
            /* Removig Temporary trace files */
            char str[255];
            std::fstream p_file;
            p_file.open(_listOfTraceFileNames.c_str());

            if (p_file.is_open())
            {
               while (!p_file.eof() )
               {
                  p_file.getline (str, 255);
                  if ( strlen(str) > 0 )
                  {
                     unsigned int i;
                     for (i = 0; i < strlen(str); i++) { if ( str[i] == ' ' ) {str[i] = 0x0; break;} }
                     if ( remove(str) != 0 ) message0("nanox: Unable to delete temporary/partial trace file" << str);
                     /* Try to remove sample file: if present */
                     str[i-4]='s';str[i-3]='a';str[i-2]='m';str[i-1]='p';str[i]='l';str[i+1]='e';str[i+2]=0x0;
                     remove(str);
                  }
               }
               p_file.close();
            }
            else message0("Unable to open " << _listOfTraceFileNames << " file");

            if ( remove(_listOfTraceFileNames.c_str()) != 0 ) message0("Unable to delete "<< _listOfTraceFileNames << " file");
         
            /* Removing EXTRAE_FINAL_DIR temporary directories and files */
            file_exists = true; num = 0;
            while ( file_exists ) {
               sprintf( file_name, "%s/set-%d", _traceFinalDirectory.c_str(), num++ );
               if ( remove( file_name ) != 0 ) file_exists = false;

            } 
            remove( _traceFinalDirectory.c_str());
         }

         /* Removing (always) EXTRAE_DIR temporary directories and files */
         file_exists = true; num = 0;
         while ( file_exists ) {
            sprintf( file_name, "%s/set-%d", _traceDirectory.c_str(), num++ );
            if ( remove( file_name ) != 0 ) file_exists = false;

         }
         remove( _traceDirectory.c_str());

      }

      void getTraceFileName ()
      {
         // Check if the trace file exists
         struct stat buffer;
         int err1, err2, err3;
         std::string file_name;
         _binFileName = ( OS::getArg( 0 ) );
         size_t found = _binFileName.find_last_of("/\\");

         /* Choose between executable name or user name */
         std::string trace_base;
         if ( _traceBaseName.compare("") != 0 ) {
            trace_base = _traceBaseName;
         } else {
            trace_base = _binFileName.substr(found+1);
         }

         if ( _skipMerge) trace_base = trace_base + "-local";

         int num = 1;
         std::string trace_suffix = "_001";
         bool file_exists = true;

         while ( file_exists ) {
            // Attempt to get the file attributes
            file_name = _traceParaverDirectory + "/" + trace_base + trace_suffix + ".prv";
            err1 = stat( file_name.c_str(), &buffer );
            file_name = _traceParaverDirectory + "/" + trace_base + trace_suffix + ".pcf";
            err2 = stat( file_name.c_str(), &buffer );
            file_name = _traceParaverDirectory + "/" + trace_base + trace_suffix + ".row";
            err3 = stat( file_name.c_str(), &buffer );

            if ( err1 == 0 || err2 == 0 || err3 == 0) {
               // Some of the files exist
               num++;
               std::stringstream trace_num;
               trace_num << "_" << (num < 100 ? "0" : "") << (num < 10 ? "0" : "") << num;
               trace_suffix =  trace_num.str();
            } else {
               std::ofstream trace_file(file_name.c_str(), std::ios::out);
               if ( !trace_file.fail() ) file_exists = false;
            }
         }

         /* New file names */
         _traceFileName_PRV = _traceParaverDirectory + "/" + trace_base + trace_suffix + ".prv" ;
         _traceFileName_PCF = _traceParaverDirectory + "/" + trace_base + trace_suffix + ".pcf";
         _traceFileName_ROW = _traceParaverDirectory + "/" + trace_base + trace_suffix + ".row";
      }

      void initialize ( void )
      {
         char *mpi_trace_on;
         char *mpi_trace_dir;
         char *mpi_trace_final_dir;
         char *tmp_dir;
         char *tmp_dir_backup;
         char *env_tmp_dir = NEW char[255];
         char *env_trace_dir = NEW char[255];
         char *env_trace_final_dir = NEW char[255];

         /* check environment variable: EXTRAE_ON */
         mpi_trace_on = getenv("EXTRAE_ON");
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

         /* check environment variable: TMPDIR */
         tmp_dir = getenv("TMPDIR");
         /* if TMPDIR defined, save it and remove it */
         if ( tmp_dir != NULL ) {
            tmp_dir_backup = NEW char[strlen(tmp_dir)];
            strcpy(tmp_dir_backup, tmp_dir);
            sprintf(env_tmp_dir, "TMPDIR=");
            putenv (env_tmp_dir);
         }
         else {
            tmp_dir_backup = NEW char[3];
            strcpy(tmp_dir_backup, "./");
         }

         recursiveMkdir(mpi_trace_dir, S_IRWXU);
         recursiveMkdir(mpi_trace_final_dir, S_IRWXU);

         /* Creating temporary directory and setting directory */
         _traceDirectory = tempnam(mpi_trace_dir, "trace");
         if ( recursiveMkdir(_traceDirectory.c_str(), S_IRWXU ) != 0 )
            fatal0 ( "Trace directory doesn't exists or user has no permissions on this directory" ); ;

         /* Creating temporary final directory and setting final directory */
         _traceFinalDirectory = tempnam(mpi_trace_final_dir, "trace");
         if ( recursiveMkdir(_traceFinalDirectory.c_str(), S_IRWXU) != 0 ) {
            remove ( _traceDirectory.c_str());
            fatal0 ( "Trace final directory doesn't exists or user has no permissions on this directory" ); ;
         }

         if ( _skipMerge ) _traceParaverDirectory = _traceFinalDirectory;
         else   _traceParaverDirectory = mpi_trace_final_dir; 

         _listOfTraceFileNames = _traceFinalDirectory + "/TRACE.mpits";

         /* Restoring TMPDIR*/
         if ( tmp_dir != NULL ) {
            sprintf(env_tmp_dir, "TMPDIR=%s", tmp_dir_backup);
            putenv (env_tmp_dir);
         }

         /* Setting EXTRAE_DIR environment variable */
         sprintf(env_trace_dir, "EXTRAE_DIR=%s", _traceDirectory.c_str());
         putenv (env_trace_dir);

         /* Setting EXTRAE_FINAL_DIR environment variable */
         sprintf(env_trace_final_dir, "EXTRAE_FINAL_DIR=%s", _traceFinalDirectory.c_str());
         putenv (env_trace_final_dir);

#ifdef NANOX_EXTRAE_DEFINE_CALLBACKS
        // Common thread information
        Extrae_set_threadid_function ( nanos_ompitrace_get_thread_num );
        Extrae_set_numthreads_function ( nanos_ompitrace_get_max_threads );

        // Cluster specific information
        void Extrae_set_taskid_function ( nanos_extrae_node_id );
        void Extrae_set_numtasks_function ( nanos_extrae_num_nodes );
        void Extrae_set_barrier_tasks_function ( nanos_ompitrace_instrumentation_barrier );
#endif

         /* OMPItrace initialization */
         OMPItrace_init();
      }

      void finalize ( void )
      {
         OMPItrace_fini();
         getTraceFileName();
         if ( !_skipMerge ) {
            mergeParaverTraceFiles();
            postProcessTraceFile();
         }
         modifyParaverConfigFile();
         modifyParaverRowFile();
         removeTemporaryFiles();
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
                  // continue...
               case NANOS_POINT:
               case NANOS_BURST_START:
               case NANOS_BURST_END:
                  ce.nEvents += e.getNumKVs();
                  break;
               default: break;
            }
         }

         ce.Types = (extrae_type_t *) alloca (ce.nEvents * sizeof (extrae_type_t));
         ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
         ce.Communications = (extrae_user_communication_t *) alloca (ce.nCommunications * sizeof ( extrae_user_communication_t));

         int j = 0; int k = 0;
         Event::ConstKVList kvs = NULL;
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
                  switch ( e.getDomain() )
                  {
                     case NANOS_WD_DOMAIN:
                     case NANOS_WD_DEPENDENCY:
                     case NANOS_WD_REMOTE:
                        ce.Communications[k].size = e.getId(); // FIXME: just in some cases size is equal to id
                        break;
                     case NANOS_XFER_PUT:
                     case NANOS_XFER_GET:
                        {
                           kvs = e.getKVs();
                           for ( unsigned int kv = 0 ; kv < e.getNumKVs() ; kv++,kvs++ ) {
                              if ( kvs->first == sizeKey ) {
                                 ce.Communications[k].size = kvs->second;
                              }
                           }
                        }
                        break;
                     default: 
                        break; // FIXME here goes a fatal
                  }
                     
                  ce.Communications[k].partner = e.getPartner();
                  k++;
                  // continue...
               case NANOS_POINT:
               case NANOS_BURST_START:
                  kvs = e.getKVs();
                  for ( unsigned int kv = 0 ; kv < e.getNumKVs() ; kv++,kvs++ ) {
                     ce.Types[j] = _eventBase + kvs->first;
                     ce.Values[j++] = kvs->second;
                  }
                  break;
               case NANOS_BURST_END:
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
#ifndef NANOX_EXTRAE_DISCARD_RESUME
          //Extrae_resume_virtual_thread ( w.getId() + nanos_ompitrace_get_max_threads() - 1 );
          Extrae_resume_virtual_thread ( w.getId() - 1 );
#endif
      }

      void addSuspendTask( WorkDescriptor &w, bool last )
      {
#ifndef NANOX_EXTRAE_DISCARD_SUSPEND
#if 0
         extrae_combined_events_t ce;
         extrae_type_t types[1] = { _eventState };
         extrae_value_t values[1] = { NANOS_NOT_RUNNING }; 

         ce.HardwareCounters = 1;
         ce.Callers = 0;
         ce.UserFunction = 0;
         ce.nEvents = 1;
         ce.nCommunications = 0;
         ce.Types = types;
         ce.Values = values;

         Extrae_emit_CombinedEvents ( &ce );
#endif

         Extrae_suspend_virtual_thread ();
#endif
      }

      void threadStart( BaseThread &thread ) {}
      void threadFinish ( BaseThread &thread ) {}
#endif
};

#ifdef NANOS_INSTRUMENTATION_ENABLED
std::string InstrumentationExtrae::_traceBaseName = std::string("");
std::string InstrumentationExtrae::_postProcessScriptPath = std::string("");
bool InstrumentationExtrae::_keepMpits = false;
bool InstrumentationExtrae::_skipMerge = false;
#endif

namespace ext {

class InstrumentationParaverPlugin : public Plugin {
   public:
      InstrumentationParaverPlugin () : Plugin("Instrumentation which generates a Paraver trace.",1) {}
      ~InstrumentationParaverPlugin () {}

      void config( Config &cfg )
      {
#ifdef NANOS_INSTRUMENTATION_ENABLED
         cfg.setOptionsSection( "Extrae module", "Extrae instrumentation module" );

         cfg.registerConfigOption ( "extrae-file-name",
                                       NEW Config::StringVar ( InstrumentationExtrae::_traceBaseName ),
                                       "Defines extrae instrumentation file name" );
         cfg.registerArgOption ( "extrae-file-name", "extrae-file-name" );
         cfg.registerEnvOption ( "extrae-file-name", "NX_EXTRAE_FILE_NAME" );

         cfg.registerConfigOption ( "extrae-post-process",
                                       NEW Config::StringVar ( InstrumentationExtrae::_postProcessScriptPath ),
                                       "Defines extrae post processing script location" );
         cfg.registerArgOption ( "extrae-post-process", "extrae-post-processor-path" );
         cfg.registerEnvOption ( "extrae-post-process", "NX_EXTRAE_POST_PROCESSOR_PATH" );
         

         cfg.registerConfigOption ( "extrae-keep-mpits", NEW Config::FlagOption( InstrumentationExtrae::_keepMpits ),
                                       "Keeps mpits temporary files generated by extrae library" );
         cfg.registerArgOption ( "extrae-keep-mpits", "extrae-keep-mpits" );

         cfg.registerConfigOption ( "extrae-skip-merge", NEW Config::FlagOption( InstrumentationExtrae::_skipMerge ),
                                       "Skips merge phase in trace generation (also keeps mpits temporary files)" );
         cfg.registerArgOption ( "extrae-skip-merge", "extrae-skip-merge" );
#endif
      }

      void init ()
      {
         sys.setInstrumentation( NEW InstrumentationExtrae() );
      }
};

} // namespace ext

} // namespace nanos

DECLARE_PLUGIN("instrumentation-paraver",nanos::ext::InstrumentationParaverPlugin);

#endif
