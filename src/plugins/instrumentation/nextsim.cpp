/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include <queue>
#include <sys/time.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>

#include <cassert>

#include "plugin.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"
#include "os.hpp"

#include <nextsim/trace/ompss/Trace.h>
#include <nextsim/trace/BinaryEventStream.h>

namespace nanos {

const unsigned int WD_ID_PHASE = 32;
const unsigned int USER_CODE = 33;
const unsigned int USER_FUNCTION = 34;
const unsigned int FLUSH = 71;

//This is because wd ids start at 1
// and we want to access the vector starting at 0
//#define WD_INDEX(__id__) (__id__-1)
//#define WD_ID(__index__) (__index__+1)
#define WD_INDEX(__id__) (__id__)
#define WD_ID(__index__) (__index__)

#if defined(NDEBUG)
#define verify(expr) expr
#else
#define verify(expr) assert(expr)
#endif

class TaskSimEvents {
private:
    std::vector<unsigned> count_vector_;

public:
    static unsigned long long       proc_timebase_mhz_;

#if defined(__x86_64__) || defined(__amd64__)
    static __inline__ unsigned long long getns(void)
    {
        unsigned long long low, high;
        asm volatile("rdtsc": "=a" (low), "=d" (high));
        return (low | (((uint64_t)high)<<32)) * 1000 / proc_timebase_mhz_;
    }
#endif
#if defined(__i386__)
    static __inline__ unsigned long long getns(void)
    {
        unsigned long long ret;
        __asm__ __volatile__("rdtsc": "=A" (ret));
        // no input, nothing else clobbered
        return ret*1000 / proc_timebase_mhz_;
    }
#endif

private:

    sim::trace::ompss::FileTrace<sim::trace::BinaryEventStream> *trace_writer_;

public:
    // constructor
    TaskSimEvents ( ) :
        trace_writer_(NULL)
     {
         FILE *fp;
         char buffer[ 32768 ];
         size_t bytes_read;
         char* match;
         double temp;
         int res;

         fp = fopen( "/proc/cpuinfo", "r" );
         bytes_read = fread( buffer, 1, sizeof( buffer ), fp );
         fclose( fp );

         if (bytes_read == 0)
           return;

         buffer[ bytes_read ] = '\0';
         match = strstr( buffer, "cpu MHz" );
         if (match == NULL)
           return;

         res = sscanf (match, "cpu MHz    : %lf", &temp );

         proc_timebase_mhz_ = (res == 1) ? temp : 0;
     }

    // destructor
    ~TaskSimEvents ( ) { }

    inline
    void init ( )
    {
        const char *out_trace = getenv("OMPSS_TRACE_FILE");
        if(out_trace == NULL) {
            std::cerr << "Please, set the OMPSS_TRACE_FILE environment variable to the path of the output trace" << std::endl;
            exit(-1);
        }
        trace_writer_ = new sim::trace::ompss::FileTrace<sim::trace::BinaryEventStream>(out_trace);
        unsigned name_id;
        verify(trace_writer_->add_task_name("__main__", name_id) == true);

        // Add the work descriptor for the main task
        trace_writer_->add_wd_info(WD_INDEX(1));
        count_vector_.push_back(name_id); // Name ID is for the main task
    }

    inline
    void fini ( )
    {
        delete trace_writer_;
    }


    inline
    void pause_wd(unsigned long long timestamp, unsigned int wd_id)
    {
       wd_id = WD_INDEX(wd_id);
       sim::trace::ompss::wd_info_t& wd = trace_writer_->get_wd_info(wd_id);

       assert(wd.active_ == true and wd.phase_stack_.empty() == false);

       //closing phase (if existed)
       if( timestamp > wd.phase_st_time_ ) {
          unsigned long long time = timestamp - wd.phase_st_time_;
          verify(trace_writer_->add_event(sim::trace::ompss::event_t(wd_id, sim::trace::ompss::PHASE_EVENT, wd.phase_stack_.top(), time)) == true);
       }

       wd.active_ = false;
       //wd.phase_stack.pop();
       //assert(wd.phase_stack.empty() == true);
    }

    inline
    void resume_wd(unsigned long long timestamp, unsigned int wd_id)
    {
       assert(wd_id > 0);
       wd_id = WD_INDEX(wd_id);
       sim::trace::ompss::wd_info_t& wd = trace_writer_->get_wd_info(wd_id);
       assert(wd.active_ == false);
       wd.active_ = true;
       if (wd.phase_stack_.empty() == false) { //not start of wd
          wd.phase_st_time_ = getns();
       }
    }
 
    inline
    void start_phase(unsigned long long timestamp, unsigned int wd_id, unsigned int phase)
    {
        wd_id = WD_INDEX(wd_id);
        sim::trace::ompss::wd_info_t& wd = trace_writer_->get_wd_info(wd_id);

        assert(wd.active_ == true and wd.phase_stack_.empty() == false);

        //close previous phase (if existed)
        if( timestamp > wd.phase_st_time_ ) {
            unsigned long long time = timestamp - wd.phase_st_time_;
            verify(trace_writer_->add_event(sim::trace::ompss::event_t(wd_id, sim::trace::ompss::PHASE_EVENT, wd.phase_stack_.top(), time)) == true);
        }

        //start counting this starting one
        wd.phase_stack_.push(phase);
        wd.phase_st_time_ = getns();
    }


    inline
    void start_wd_id(unsigned long long timestamp, unsigned int wd_id, unsigned int phase)
    {
        assert(wd_id > 0);
        wd_id = WD_INDEX(wd_id);
        sim::trace::ompss::wd_info_t& wd = trace_writer_->get_wd_info(wd_id);
        assert(wd.active_ == true);
        wd.active_ = true;
        assert(wd.phase_stack_.empty() == true);

        wd.phase_stack_.push(phase);
        wd.phase_st_time_ = getns();
    }


    inline
    void stop_phase(unsigned long long timestamp, unsigned int wd_id, unsigned int phase, const std::string& name = "")
    {
        wd_id = WD_INDEX(wd_id);
        sim::trace::ompss::wd_info_t& wd = trace_writer_->get_wd_info(wd_id);
        assert(wd.active_ == true and wd.phase_stack_.empty() == false);
        assert(wd.phase_stack_.top() == phase);
        // Closing phase (if existed)

        if( timestamp > wd.phase_st_time_ ) {
            unsigned long long time = getns() - wd.phase_st_time_;

            if(name.empty() == false) {
                unsigned name_id;
                unsigned name_count = 0;
                if(trace_writer_->add_task_name(name, name_id) == true) { /* This is a new name */
                    assert(count_vector_.size() == name_id);
                    count_vector_.push_back(0);
                    name_count = 0;
                }
                else {
                    name_count = count_vector_[name_id];
                }

                verify(trace_writer_->add_event(sim::trace::ompss::event_t(wd_id, sim::trace::ompss::PHASE_EVENT, phase, time, name_id, name_count)) == true);
                count_vector_[name_id]++;
             }
             else {
                verify(trace_writer_->add_event(sim::trace::ompss::event_t(wd_id, sim::trace::ompss::PHASE_EVENT, phase, time)) == true);
             }
         }

         wd.phase_stack_.pop();
         assert(wd.phase_stack_.empty() == false);
         wd.phase_st_time_ = getns();
    }


    inline
    void stop_wd_id(unsigned long long timestamp, unsigned int wd_id, unsigned int phase)
    {
        wd_id = WD_INDEX(wd_id);
        sim::trace::ompss::wd_info_t& wd = trace_writer_->get_wd_info(wd_id);

        assert(wd.active_ == true and wd.phase_stack_.empty() == false);
        assert(wd.phase_stack_.top() == phase);

        //closing phase (if existed)
        if( timestamp > wd.phase_st_time_ ) {
            unsigned long long time = timestamp - wd.phase_st_time_;
            verify(trace_writer_->add_event(sim::trace::ompss::event_t(wd_id, sim::trace::ompss::PHASE_EVENT, wd.phase_stack_.top(), time)) == true);
        }

        wd.active_ = false;
        wd.phase_stack_.pop();
        assert(wd.phase_stack_.empty() == true);
    }


    inline
    void wd_creation(unsigned int wd_id, int num_deps, nanos_data_access_t* dep)
    {
        //new wd in the map
        wd_id = WD_INDEX(wd_id);
        sim::trace::ompss::wd_info_t &new_wd = trace_writer_->add_wd_info(wd_id);
        new_wd.active_ = false;

        for(int i = 0; i < num_deps; i++) {
            new_wd.deps_.push_back(dep[i]);
        }
        new_wd.phase_st_time_ = getns();
    }


    inline
    void add_event(unsigned long long timestamp, unsigned int wd_id, unsigned int type, unsigned int val1, unsigned int val2)
    {
        wd_id = WD_INDEX(wd_id);
        sim::trace::ompss::wd_info_t& wd = trace_writer_->get_wd_info(wd_id);
        assert(wd.active_ == true and wd.phase_stack_.empty() == false);

        //close current phase (if existed) and re-open it after new event
        if( timestamp > wd.phase_st_time_ ) {
            unsigned long long time = timestamp - wd.phase_st_time_;
            verify(trace_writer_->add_event(sim::trace::ompss::event_t(wd_id, sim::trace::ompss::PHASE_EVENT, wd.phase_stack_.top(), time)) == true);
        }
        verify(trace_writer_->add_event(sim::trace::ompss::event_t(wd_id, static_cast<sim::trace::ompss::type_t>(type), val1, val2)) == true);

        wd.phase_st_time_ = getns();
    }


    inline
    void add_dep(unsigned wd_id, unsigned int num_deps, nanos_data_access_t* dep)
    {
        for(unsigned int i = 0; i < num_deps; i++) {
            trace_writer_->add_dep({wd_id, dep[i]});
        }
    }

};

unsigned long long TaskSimEvents::proc_timebase_mhz_;

class InstrumentationTasksimTrace: public Instrumentation
{
#ifndef NANOS_INSTRUMENTATION_ENABLED
   public:
      // constructor
      InstrumentationTasksimTrace(): Instrumentation() { }
      // destructor
      virtual ~InstrumentationTasksimTrace() {}

      // low-level instrumentation interface (mandatory functions)
      virtual void initialize( void ) {}
      virtual void finalize( void ) {}
      virtual void threadStart ( BaseThread &thread ) {}
      virtual void threadFinish ( BaseThread &thread ) {}
      virtual void enable() {}
      virtual void disable() {}
      virtual void addResumeTask(nanos::WorkDescriptor&) {}
      virtual void addSuspendTask(nanos::WorkDescriptor&, bool) {}
      virtual void addEventList ( unsigned int count, Event *events ) {}
#else
   private:
       TaskSimEvents      wd_events_;

       unsigned int phase_id_offset_;

       inline unsigned int getMyWDId()
       {
           BaseThread *current_thread = getMyThreadSafe();
           if(current_thread == NULL) return 0;
           else if(current_thread->getCurrentWD() == NULL) return 0;
           return current_thread->getCurrentWD()->getId();
       }

   public:
       // constructor
       InstrumentationTasksimTrace (): Instrumentation( *new InstrumentationContextDisabled() ) { }

      // destructor
      ~InstrumentationTasksimTrace ( ) { }

      // low-level instrumentation interface (mandatory functions)

      virtual void initialize( void )
      {
          wd_events_.init();
      }
      virtual void finalize( void )
      {
          wd_events_.fini();
      }

      virtual void enable() { std::cerr << "Disabling tasksim instrumentation not supported" << std::endl; exit(1); }
      virtual void disable() { std::cerr << "Disabling tasksim instrumentation not supported" << std::endl; exit(1); }

      virtual void addResumeTask(nanos::WorkDescriptor& wd) { wd_events_.resume_wd(TaskSimEvents::getns(), wd.getId()); }
      virtual void addSuspendTask(nanos::WorkDescriptor& wd, bool) { wd_events_.pause_wd(TaskSimEvents::getns(), wd.getId()); }

      virtual void addEventList ( unsigned int count, Event *events )
      {
          static const nanos_event_key_t api          = getInstrumentationDictionary()->getEventKey("api");
          static const nanos_event_value_t wait_group = getInstrumentationDictionary()->getEventValue("api","wg_wait_completion");
          static const nanos_event_key_t wd_id        = getInstrumentationDictionary()->getEventKey("wd-id");
          static const nanos_event_key_t user_func    = getInstrumentationDictionary()->getEventKey("user-funct-location");
          static const nanos_event_key_t user_code    = getInstrumentationDictionary()->getEventKey("user-code");

          static const nanos_event_key_t create_wd_id = getInstrumentationDictionary()->getEventKey("create-wd-id");
          static const nanos_event_key_t wd_num_deps  = getInstrumentationDictionary()->getEventKey("wd-num-deps");
#ifndef NDEBUG
          static const nanos_event_key_t wd_ptr       = getInstrumentationDictionary()->getEventKey("create-wd-ptr");
          static const nanos_event_key_t wd_deps_ptr  = getInstrumentationDictionary()->getEventKey("wd-deps-ptr");
#endif

         unsigned int which_WD = getMyWDId();

         unsigned long long timestamp = TaskSimEvents::getns();

         for (unsigned int i = 0; i < count; i++) {
            Event &e = events[i];
            unsigned int type = e.getType();

            switch ( type ) {
               case NANOS_STATE_START:
               case NANOS_STATE_END:
               case NANOS_SUBSTATE_START:
               case NANOS_SUBSTATE_END:
               case NANOS_PTP_START:
               case NANOS_PTP_END:
                  break;
               case NANOS_POINT: {
                  //We recognize the point event class through the first key type
                  if ( e.getKey() == create_wd_id ) { //It's a task creation event
                      unsigned int wd_id_num = e.getValue();
                      wd_events_.add_event(timestamp, which_WD, sim::trace::ompss::CREATE_TASK_EVENT, wd_id_num, 0);
                      e = events[++i];
                      assert(e.getKey() == wd_ptr);
                      //WD pointer is not used for now, but it may be useful for future extensions
                      //nanos_wd_t* wd = (nanos_wd_t*) e.getValue();
                      e = events[++i];
                      assert(e.getKey() == wd_num_deps);
                      int num_deps =  e.getValue();
                      e = events[++i];
                      assert(e.getKey() == wd_deps_ptr);
                      wd_events_.wd_creation(wd_id_num, num_deps, (nanos_data_access_t*) e.getValue());
                  }
                  else if ( e.getKey() == wd_num_deps ) { //It's a wait on event
                      int num_deps = e.getValue();
                      e = events[++i];
                      assert(e.getKey() == wd_deps_ptr);
                      wd_events_.add_event(timestamp, which_WD, sim::trace::ompss::WAIT_ON_EVENT, num_deps, 0);
                      wd_events_.add_dep(which_WD, num_deps, (nanos_data_access_t*) e.getValue());
                  } else {
                      //ignore any other point event
                  }
                  break;
               }
               case NANOS_BURST_START: {
                  nanos_event_key_t e_key = e.getKey();
                  nanos_event_key_t e_val = e.getValue();
                  if ( e_key == api )  {
                      wd_events_.start_phase(timestamp, which_WD, e_val);
                  }
                  else if ( e_key == user_code ) {
                      which_WD = e_val;
                      wd_events_.start_phase(timestamp, which_WD, USER_CODE);
                  }
                  else if ( e_key == wd_id ) {
                      which_WD = e_val;
                      wd_events_.start_wd_id(timestamp, which_WD, WD_ID_PHASE);
                  }
                  else if ( e_key == user_func ) {
                      wd_events_.start_phase(timestamp, which_WD, USER_FUNCTION);
                  }
                  break;
               }
               case NANOS_BURST_END: {
                  nanos_event_key_t e_key = e.getKey();
                  nanos_event_key_t e_val = e.getValue();
                  if ( e_key == api ) {
                     if(e_val == wait_group) { //at the end of wait group phase, add a wait group event
                        wd_events_.add_event(timestamp, which_WD, sim::trace::ompss::WAIT_GROUP_EVENT, 0, 0);
                      }
                      wd_events_.stop_phase(timestamp, which_WD, e_val);
                  } else if ( e_key == user_code ) {
                      which_WD = e_val;
                      wd_events_.stop_phase(timestamp, which_WD, USER_CODE);
                  } else if ( e_key == wd_id ) {
                      which_WD = e_val;
                      wd_events_.stop_wd_id(timestamp, which_WD, WD_ID_PHASE);
                  } else if ( e_key == user_func ) {
                      std::string name;
                      {//FIXME: whenever the new getValueKey method is available, replace this by a call to it
                         InstrumentationDictionary::ConstKeyMapIterator k_it  = getInstrumentationDictionary()->beginKeyMap();
                         InstrumentationDictionary::ConstKeyMapIterator k_end = getInstrumentationDictionary()->endKeyMap();
                         bool k_found = false, v_found = false;
                         while( k_it != k_end and not k_found) {
                            if( k_it->second->getId() == e_key ) {
                               InstrumentationKeyDescriptor::ConstValueMapIterator v_it  = k_it->second->beginValueMap();
                               InstrumentationKeyDescriptor::ConstValueMapIterator v_end = k_it->second->endValueMap();
                               while( v_it != v_end and not v_found ) {
                                  if( v_it->second->getId() == e_val ) {
                                     name = v_it->first;
                                     v_found = true;
                                  }
                                  v_it++;
                               }
                               k_found = true;
                            }
                            k_it++;
                         }
                         assert(v_found == true);
                         name = name.substr(0, name.find_first_of(":"));
                      }
                      wd_events_.stop_phase(timestamp, which_WD, USER_FUNCTION, name);
                   }
                   break;
               }
               default:
                  break;
            }
         }
      }

       virtual void threadStart ( BaseThread &thread ) {}
       virtual void threadFinish ( BaseThread &thread ) {}
#endif
};


namespace ext {

class InstrumentationTasksimTracePlugin : public Plugin {
   public:
       InstrumentationTasksimTracePlugin () : Plugin("Instrumentor which generates tasksim traces.",1) {}
      ~InstrumentationTasksimTracePlugin () {}

      void config( Config &cfg ) {}

      void init ()
      {
         sys.setInstrumentation( new InstrumentationTasksimTrace() );
         sys.getThreadManagerConf().setUseYield(true);
      }
};

} // namespace ext
} // namespace nanos

DECLARE_PLUGIN( "placeholder-name", nanos::ext::InstrumentationTasksimTracePlugin );
