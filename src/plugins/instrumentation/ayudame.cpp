#include "plugin.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"
#include "Ayudame.h"
#include "smpdd.hpp" // FIXME: this the include should not be here (just testing smpdd)

namespace nanos {

class InstrumentationAyudame: public Instrumentation 
{
#ifndef NANOS_INSTRUMENTATION_ENABLED
   public:
      // constructor
      InstrumentationAyudame() : Instrumentation() {}
      // destructor
      ~InstrumentationAyudame() {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void ) {}
      void finalize( void ) {}
      void disable( void ) {}
      void enable( void ) {}
      void addResumeTask( WorkDescriptor &w ) {}
      void addSuspendTask( WorkDescriptor &w, bool last ) {}
      void addEventList ( unsigned int count, Event *events ) {}
      void threadStart( BaseThread &thread ) {}
      void threadFinish ( BaseThread &thread ) {}
#else
   public:
      // constructor
      InstrumentationAyudame() : Instrumentation( *NEW InstrumentationContextDisabled() ) {}
      // destructor
      ~InstrumentationAyudame ( ) {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void )
      {
         ayu_runtime_t ayu_rt = 3; // FIXME: hard coded, use AYU_RT_OMPSS?
         AYU_event(AYU_PREINIT, 0, (void *) &ayu_rt);
fprintf(stderr,"Event PREINIT emitted\n");
         int64_t AYU_data[3];
         AYU_data[0] = 0; // FIXME: Function Id
         AYU_data[1] = 0; // FIXME: Is Critical?

#ifdef GPU_DEV
         const int system_tasks = sys.getNumPEs() + nanos::ext::GPUConfig::getGPUCount();
#else
         const int system_tasks = sys.getNumPEs();
#endif
         int i;

         for ( i = 1; i <= system_tasks;i++){
            AYU_event(AYU_ADDTASK, (int64_t) i, AYU_data );
fprintf(stderr,"Emitting Event (sys) ADDTASK with data={%d,%d}, id=%d\n",(int)AYU_data[0], (int)AYU_data[1], (int) i );
         }
      }
      void finalize( void ) 
      {
         AYU_event(AYU_FINISH, 0, NULL);
fprintf(stderr,"Event FINISH emitted\n");
      }
      void disable( void ) {}
      void enable( void ) {}
      void addResumeTask( WorkDescriptor &w )
      {
fprintf(stderr,"Emitting Event RUNTASK %d\n",(int) w.getId());
         AYU_event (AYU_RUNTASK, (int64_t) w.getId(), NULL);
      }
      void addSuspendTask( WorkDescriptor &w, bool last )
      {
         if (last) {
fprintf(stderr,"Emitting Event REMOVETASK %d\n",(int) w.getId());
            AYU_event(AYU_REMOVETASK, (int64_t) w.getId(), NULL);
         }
      }
      void addEventList ( unsigned int count, Event *events )
      {
         InstrumentationDictionary *iD = sys.getInstrumentation()->getInstrumentationDictionary();
         nanos_event_key_t create_task = iD->getEventKey("create-wd-ptr");
         nanos_event_key_t dependence  = iD->getEventKey("dependence");
         nanos_event_key_t key;
  
         int64_t AYU_data[3];

         for (unsigned int i = 0; i < count; i++)
         {
            Event &e = events[i];
            switch ( e.getType() ) {
               case NANOS_STATE_START:
               case NANOS_STATE_END:
               case NANOS_SUBSTATE_START:
               case NANOS_SUBSTATE_END:
               case NANOS_PTP_START:
               case NANOS_PTP_END:
                  break;
               case NANOS_POINT:
                  key = e.getKey();
                  if ( key == create_task ) {
                     WorkDescriptor *wd = (WorkDescriptor *) e.getValue();
                     int64_t wd_id = wd->getId();

#if 0
                     AYU_data[0] = 0; // FIXME: Function Id
#else
                     int64_t funct_id = (int64_t) ((ext::SMPDD &)(wd->getActiveDevice())).getWorkFct();
                     AYU_data[0] = funct_id;
                     AYU_event(AYU_REGISTERFUNCTION, funct_id, (void *) "hola");
#endif
                     AYU_data[1] = 0; // FIXME: Is Critical?
fprintf(stderr,"Emitting Event ADDTASK with data={%d,%d}, id=%d\n",(int)AYU_data[0], (int)AYU_data[1], (int) wd_id );
                     AYU_event(AYU_ADDTASK, wd_id, AYU_data );
                  } else if ( key == dependence ) {
                     nanos_event_value_t dependence_value = e.getValue();
                     int sender_id = (int) ( dependence_value >> 32 );
                     int receiver_id = (int) ( dependence_value & 0xFFFFFFFFFF );
                     AYU_data[0] = sender_id; // 
                     AYU_data[1] = 0; // FIXME: mem addr
                     AYU_data[2] = 0; // FIXME: mem addr
fprintf(stderr,"Emitting Event ADDDEPENDENCY between %d and %d, mem_addr ={%d,%d}\n",(int) AYU_data[0], (int) receiver_id, (int) AYU_data[1], (int) AYU_data[2]);
                     AYU_event(AYU_ADDDEPENDENCY, (int64_t) receiver_id, AYU_data );
                  }
                  break;
               case NANOS_BURST_START:
               case NANOS_BURST_END:
               default: break;
            }
         }

      }
      void threadStart( BaseThread &thread ) {}
      void threadFinish ( BaseThread &thread ) {}
#endif

};

namespace ext {

class InstrumentationAyudamePlugin : public Plugin {
   public:
      InstrumentationAyudamePlugin () : Plugin("Instrumentation which implements Ayudame/Temanejo protocol.",1) {}
      ~InstrumentationAyudamePlugin () {}

      void config( Config &cfg ) {}

      void init ()
      {
         sys.setInstrumentation( new InstrumentationAyudame() );	
      }
};

} // namespace ext

} // namespace nanos

DECLARE_PLUGIN("intrumentation-ayudame",nanos::ext::InstrumentationAyudamePlugin);
