#include "plugin.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"
#include "ompt_callbacks.h"

extern "C" {

   int ompt_initialize(
         ompt_function_lookup_t lookup,  /* function to look up OMPT API routines by name */
         const char *runtime_version,    /* OpenMP runtime version string */
         unsigned int ompt_version       /* integer that identifies the OMPT revision */
         );

   int ompt_initialize( ompt_function_lookup_t lookup, const char *runtime_version, unsigned int ompt_version )
   {
      fatal( "There is no OMPT compliant tool loaded\n" );
      return 0;
   } 


   //! List of callback declarations
   ompt_new_parallel_callback_t ompt_nanos_event_parallel_begin = NULL;
   ompt_parallel_callback_t ompt_nanos_event_parallel_end = NULL;
   ompt_new_task_callback_t     ompt_nanos_event_task_begin = NULL;
   ompt_task_callback_t         ompt_nanos_event_task_end = NULL;
   ompt_thread_type_callback_t  ompt_nanos_event_thread_begin = NULL;
   ompt_thread_type_callback_t  ompt_nanos_event_thread_end = NULL;

   int ompt_nanos_set_callback( ompt_event_t event, ompt_callback_t callback );
   int ompt_nanos_set_callback( ompt_event_t event, ompt_callback_t callback )
   {
      switch ( event ) {
         case ompt_event_parallel_begin:
            ompt_nanos_event_parallel_begin = (ompt_new_parallel_callback_t) callback;
            return 4;
         case ompt_event_parallel_end:
            ompt_nanos_event_parallel_end = (ompt_parallel_callback_t) callback;
            return 4;
         case ompt_event_task_begin: 
            ompt_nanos_event_task_begin = (ompt_new_task_callback_t) callback;
            return 4;
         case ompt_event_task_end:
            ompt_nanos_event_task_end = (ompt_task_callback_t) callback;
            return 4;
         case ompt_event_thread_begin:
            ompt_nanos_event_thread_begin = (ompt_thread_type_callback_t) callback;
            return 4;
         case ompt_event_thread_end:
            ompt_nanos_event_thread_end = (ompt_thread_type_callback_t) callback;
            return 4;
         case ompt_event_control:
            return 4;
         case ompt_event_runtime_shutdown:
            return 4;
         default:
            warning("Callback registration error");
            return 0;
      }

   }

   ompt_interface_fn_t ompt_nanos_lookup ( const char *entry_point );
   ompt_interface_fn_t ompt_nanos_lookup ( const char *entry_point )
   {
      if ( strncmp( entry_point, "ompt_set_callback", strlen("ompt_set_callback") ) == 0 )
         return ( ompt_interface_fn_t ) ompt_nanos_set_callback;
 
      return (NULL);
   }
}

namespace nanos
{
   class InstrumentationOMPT: public Instrumentation 
   {
      public:
         InstrumentationOMPT( ) : Instrumentation( *NEW InstrumentationContextDisabled() ) {}
         ~InstrumentationOMPT() {}
         void initialize( void )
         {
            ompt_initialize ( ompt_nanos_lookup, "Nanos++ 0.8a", 1);
         }
         void finalize( void ) {}
         void disable( void ) {}
         void enable( void ) {}
         void addEventList ( unsigned int count, Event *events )
         {
            InstrumentationDictionary *iD = getInstrumentationDictionary();
            static const nanos_event_key_t create_wd_ptr = iD->getEventKey("create-wd-ptr");
            static const nanos_event_key_t api = iD->getEventKey("api");
            static const nanos_event_value_t api_enter_team = iD->getEventValue("api","enter_team");
            static const nanos_event_value_t api_leave_team = iD->getEventValue("api","leave_team");

            unsigned int i;
            for( i=0; i<count; i++) {
               Event &e = events[i];
               if ( e.getKey( ) == create_wd_ptr && ompt_nanos_event_task_begin )
               { 
                  ompt_nanos_event_task_begin((ompt_task_id_t) nanos::myThread->getCurrentWD()->getId(), NULL,
                                              (ompt_task_id_t) ((WorkDescriptor *)e.getValue())->getId(), NULL);
               }
               else if ( e.getKey( ) == api )
               {
                  nanos_event_value_t val = e.getValue();
                
                  if ( val == api_enter_team && ompt_nanos_event_parallel_begin )
                  {
                     ompt_nanos_event_parallel_begin (
                       (ompt_task_id_t) nanos::myThread->getCurrentWD()->getId(),
                       (ompt_frame_t) NULL,
                       (ompt_parallel_id_t) 0,
                       (uint32_t) 0,
                       (void *) NULL );
                  }
                  else if ( val == api_leave_team && ompt_nanos_event_parallel_end )
                  {
                     ompt_nanos_event_parallel_end (
                           (ompt_parallel_id_t) 0,
                           (ompt_task_id_t) nanos::myThread->getCurrentWD()->getId() );
                  }
          
               }
            }
         }
         void addResumeTask( WorkDescriptor &w ) {}
         void addSuspendTask( WorkDescriptor &w, bool last )
         {
            if (ompt_nanos_event_task_end && last) ompt_nanos_event_task_end((ompt_task_id_t) w.getId());
         }
         void threadStart( BaseThread &thread ) 
         {
            if (ompt_nanos_event_thread_begin) ompt_nanos_event_thread_begin((ompt_thread_type_t) 2, (ompt_thread_id_t) nanos::myThread->getId());
         }
         void threadFinish ( BaseThread &thread )
         {
            if (ompt_nanos_event_thread_end) ompt_nanos_event_thread_end((ompt_thread_type_t) 2, (ompt_thread_id_t) nanos::myThread->getId());
         }
         void incrementMaxThreads( void ) {}
   };
   namespace ext
   {
      class InstrumentationOMPTPlugin : public Plugin
      {
         public:
            InstrumentationOMPTPlugin () : Plugin("Instrumentation OMPT compatible.",1) {}
            ~InstrumentationOMPTPlugin () {}
            void config( Config &cfg ) {}
            void init () { sys.setInstrumentation( NEW InstrumentationOMPT() ); }
      };
   } // namespace ext
} // namespace nanos

DECLARE_PLUGIN("instrumentation-ompt",nanos::ext::InstrumentationOMPTPlugin);
