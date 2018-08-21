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

//! \file instrumentation_decl.hpp
//! \brief Instrumentation main classes declaration.
//
//! \defgroup core_instrumentation Instrumentation module
//! \ingroup core

/*!\page    core_instrumentation
 * \ingroup core_instrumentation
 *
 * \section introduction Introduction
 *
 * The main goal of instrumentation is to get some information about the program execution. In other words, we want to know "What happens in this WorkDescriptor? running on this Thread". There are the three main components involved in the instrumentation process: What (we also call it Event), WorkDescriptor and Thread.
 *
 * - Events are something that happens at a given time or at a given interval of time.
 * - WorkDescriptors are the runtime basic unit of work. They offer a context to execute a piece of code.
 * - Threads are logical (or virtual) processors that execute WorkDescriptors.
 *
 * Instrumentation defines an interface which allow to know specific details about the execution of any program using Nanos++ Runtime Library. In order to do that we have defined several concepts which represents different activities happening during the execution. Nanos++ defines four different type of events
 *
 * - Burst: a burst is defined by a time interval. During this interval something is happening (e.g. executing a runtime service).
 * - State: a state is also defined by an interval of time and defines which is the execution status in a specific time stamp. It can be considered as a specific case of burst object but with a predetermined behaviour. All state's changes are push and pop operation. So when we are changing state we are whether pushing a new state in the state stack or returning to the previous state (the one previous current state). Usually a state defines the nature of the code we are executing. We can be executing a synchronization operation (SYNCH), or waiting for more job to execute (IDLE), or useful code for the user (RUNNING), etc
 * - Point: a point event is defined by a timestamp. This entity represents a punctual event during the execution.
 * - Point-to-point: a point-to-point event is defined for two punctual events. One is called the origin and the other one destination. With these kind of events we can represent communication (send/receive procedures), or work spawning (producer/consumer schemes), etc
 *
 * Instrumentation is also driven through Key/Value pairs in which the item Key identifies the semantic of the associated Value (e.g., WorkDescriptor ID as a Key and a numerical identifier as the associated Value). Keys and Values can be registered in a global dictionary (InstrumentationDictionary) which can be used as a repository.
 *
 * In order to create and raise events from user's code see \ref capi_instrument
 *
 * \section implementation Implementation
 *
 * Instrumentation mechanism implementation is divided in several classes:
 *
 * - Instrumentation class is a singleton object which can be accessed through the - also singleton - System object (sys).
 * - InstrumentationDictionary class which is part of the Instrumentation class.
 * - InstrumentationContext class, each WorkDescriptor? object has an associated InstrumentationContext object.
 * - Instrumentation modules, help programmers in the duty of open/close events.
 *
 * In this section we will describe the instrumentation mechanism by describing the implementation of each one of these classes.
 *
 * \subsection instrumentation Instrumentation class
 * \copydoc nanos::Instrumentation
 *
 * \subsection instrumentation_context InstrumentationContext class
 * \copydoc nanos::InstrumentationContext
 *
 * \subsection instrumentation_dictionary InstrumentationDictionary class
 * \copydoc nanos::InstrumentationDictionary
 *
 * \subsection instrumentation_modules InstrumentationModules classes
 *
 * Instrumentation modules help programmers in the instrumentation process by doing automatically some of the duties that users need to follow for correct instrumentation. So far its main utility is to take care about multiple exits in a given piece of code. As a module is a C++ object, we can use the constructor to open an instrumentation burst leaving the responsibility of closing it to the corresponding destructor.
 *
 * Creating a new InstrumentState object will produce the opening of a State event (the value is specified in the object constructor). Once the object goes out of the scope where is declared the destructor will close it (if programmer has not closed it before). As most of instrumentation phases affect a whole function the programmer has just to create an object of a Instrumentation module at the beginning of the function.
 *
 * - \copydoc nanos::InstrumentStateAndBurst
 * - \copydoc nanos::InstrumentState
 * - \copydoc nanos::InstrumentBurst
 *
 * \subsection examples Instrumentation examples
 *
 * In this section we will explain how different parts of the runtime have been instrumented. As one of the design principles was to encapsulate the code and to avoid that performance runtime version has any impact by the instrumentation code (or at least keep the impact as low as possible), the runtime offers a macro which allow to remove the code when it is not needed. The NANOS_INSTRUMENT(code) macro:
 * 
 * \code
 * #ifdef NANOS_INSTRUMENTATION_ENABLED
 *    #define NANOS_INSTRUMENT(f) f;
 * #else
 *    #define NANOS_INSTRUMENT(f) ;
 * #endif
 * \endcode
 *
 * All instrumentation calls have to be protected using this macro.
 * 
 * \subsubsection example1 Example 1: Memory allocation
 * 
 * Some runtime chunks of code are bounded by instrumentation events in order to measure the duration of this piece of code. An example is a cache allocation. This function is bounded by a state event and a burst event. State event will change the current thread's state to CACHE and the Burst event will keep information of the memory allocation size for the specific call. Here is the example:
 * 
 * \code
 * void * allocate( size_t size )
 * {
 *       void *result;
 *       NANOS_INSTRUMENT(nanos_event_key_t k);
 *       NANOS_INSTRUMENT(k = Instrumentor->getInstrumentorDictionary()->getEventKey("cache-malloc"));
 *       NANOS_INSTRUMENT(Instrumentor()->raiseOpenStateAndBurst(CACHE, k, (nanos_event_value_t) size));
 *       result = _T::allocate(size);
 *       NANOS_INSTRUMENT(Instrumentor()->raiseCloseStateAndBurst(k,0));
 *       return result;
 * }
 * \endcode
 *
 * \subsubsection example2 Example 2: WorkDescriptor?'s context switch
 * 
 * WorkDescriptor?'s context switch uses two instrumentation services wdLeaveCPU() and wdEnterCPU(). The wdLeaveCPU() is called from the leaving task context execution and wdEnterCPU() is called once we are executing the new task.
 * 
 * \code
 *    .
 *    .
 *    .
 *    NANOS_INSTRUMENT( sys.getInstrumentor()->wdLeaveCPU(oldWD) );
 *    myThread->switchHelperDependent(oldWD, newWD, arg);
 * 
 *    myThread->setCurrentWD( *newWD );
 *    NANOS_INSTRUMENT( sys.getInstrumentor()->wdEnterCPU(newWD) );
 *    .
 *    .
 *    .
 * \endcode
 *
 * \subsubsection example3 Example 3: Instrumenting the API
 * 
 * API functions have – generally – a common behaviour. They open a Burst event with a pair <key,value>. The key is the internal code “api” and the value is an specific identifier of the function we are instrumenting on. API functions also open a State event with a value according with the function duty. Both events will be closed once the function execution finishes. Here it is an example using nanos_yield() implementation:
 * 
 * \code
 * nanos_err_t nanos_yield ( void )
 * {
 *    NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","yield",SCHEDULING) );
 *    try {
 *       Scheduler::yield();
 *    } catch ( ... ) {
 *       return NANOS_UNKNOWN_ERR;
 *    }
 *    return NANOS_OK;
 * }
 * \endcode
 *
 * Yield function will wrap its execution between <”api”,“yield”> Burst and SCHEDULING State events. Although the function can have other exit points (apart from the return) InstrumentStateAndBurst destructor will throw closing events automatically.
 * 
 * \subsubsection example4 Example 4: Instrumenting Runtime Internal Functions
 * 
 * Different Nanos++ functions have different instrumentation approaches. In this sections we have chosen a scheduling related function: Scheduler::waitOnCondition(). Due space limitations we have abridged the code focusing our interest in the instrumentation parts.
 * 
 * \code
 * void Scheduler::waitOnCondition (GenericSyncCond *condition)
 * {
 *    NANOS_INSTRUMENT( InstrumentState inst(SYNCHRONIZATION) );
 * 
 *    const int nspins = sys.getSchedulerConf().getNumSpins();
 *    int spins = nspins;
 * 
 *    WD * current = myThread->getCurrentWD();
 * 
 *    while ( !condition->check() ) {
 *       BaseThread *thread = getMyThreadSafe();
 *       spins--;
 *       if ( spins == 0 ) {
 *          condition->lock();
 *          if ( !( condition->check() ) ) {
 *             condition->addWaiter( current );
 * 
 *             NANOS_INSTRUMENT( InstrumentState inst1(SCHEDULING) );
 *             WD *next = _schedulePolicy.atBlock( thread, current );
 *             NANOS_INSTRUMENT( inst1.close() );
 * 
 *             if ( next ) {
 *                NANOS_INSTRUMENT( InstrumentState inst2(RUNTIME) );
 *                switchTo ( next );
 *             }
 *             else {
 *                condition->unlock();
 *                NANOS_INSTRUMENT( InstrumentState inst3(YIELD) );
 *                thread->yield();
 *             }
 *          } else {
 *             condition->unlock();
 *          }
 *          spins = nspins;
 *       }
 *    }
 * }
 * \endcode
 *
 * In this function the instrumentation changes the thread state in several parts of the code. First, all the function code is surrounded by a SYNCHRONIZATION state (inst). A Opening state event is raised at the very beginning of the function and the corresponding close event will be thrown once the execution flow gets out from the function scope. During the function execution the thread state may change to SCHEDULING when calling _schedulePolicy.atBlock(), RUNTIME when we are context switching WorkDescriptors? and YIELD when we are forcing a thread yield. In this case the SCHEDULING state change is the only one we have to force to close before getting out from its scope. Note, that if an C++ exception is raised by any of the lower layers the states that are open at point will close automatically. So, the use of the Instrumentation modules improves the general exception safety of the code.
 * 
 */

#ifdef NANOS_INSTRUMENTATION_ENABLED
#define NANOS_INSTRUMENT(f) f;
#else
#define NANOS_INSTRUMENT(f)
#endif

#ifndef __NANOS_INSTRUMENTOR_DECL_H
#define __NANOS_INSTRUMENTOR_DECL_H
#include <list>
#include <utility>
#include <string>
#include "compatibility.hpp"
#include "debug.hpp"
#include "nanos-int.h"
#include "atomic_decl.hpp"
#include "lock_decl.hpp"
#include "instrumentationcontext_fwd.hpp"
#include "workdescriptor_fwd.hpp"
#include "allocator_decl.hpp"
#include "basethread_fwd.hpp"


#define NANOX_INSTRUMENTATION_PARTNER_MYSELF 0xFFFFFFFF

namespace nanos {

   // This is an ordered list. The idea is to emit all events below or equal the event category
   // predetermined.
   typedef enum { EVENT_ENABLED = 0, EVENT_NONE, EVENT_USER, EVENT_DEVELOPER,
                  EVENT_DEFAULT, EVENT_ADVANCED, EVENT_ALL, EVENT_DISABLED } nanos_event_level_t;

//   extern __thread BaseThread *myThread;
#ifdef NANOS_INSTRUMENTATION_ENABLED
   class InstrumentationValueDescriptor
   {
      private:
         nanos_event_value_t  _id;          /**< InstrumentationValueDescriptor id */
         std::string          _description; /**< InstrumenotrValueDescriptor description */
      private:
         /*! \brief InstrumentationValueDescriptor default constructor (private)
          */
         InstrumentationValueDescriptor ();
         /*! \brief InstrumentationValueDescriptor copy constructor (private)
          */
         InstrumentationValueDescriptor ( InstrumentationValueDescriptor &ivd );
         /*! \brief InstrumentationValueDescriptor copy assignment operator (private)
          */
         InstrumentationValueDescriptor& operator= ( InstrumentationValueDescriptor &ivd );
      public:
         /*! \brief InstrumentationValueDescriptor constructor
          */
         InstrumentationValueDescriptor ( nanos_event_value_t id, const std::string &description ) : _id( id ), _description ( description ) {}

         /*! \brief InstrumentationValueDescriptor constructor
          */
         InstrumentationValueDescriptor ( nanos_event_value_t id, const char *description ) : _id( id ), _description ( description ) {}

         /*! \brief InstrumentationValueDescriptor destructor
          */
         ~InstrumentationValueDescriptor() {}

         /*! \brief Gets value descriptor id
          */
         nanos_event_value_t getId ( void );

         /*! \brief Gets value descriptor textual description
          */
         const std::string getDescription ( void ) const;

   };

   class InstrumentationKeyDescriptor
   {
      public:
         typedef TR1::unordered_map<std::string, InstrumentationValueDescriptor*> ValueMap;
         typedef ValueMap::iterator ValueMapIterator;
         typedef ValueMap::const_iterator ConstValueMapIterator;
      private:
         nanos_event_key_t    _id;          /**< InstrumentationKeyDescriptor id */
         nanos_event_level_t  _level;       /**< Event level of verbosity */
         bool                 _stacked;     /**< Is the event stacked */
         std::string          _description; /**< InstrumenotrKeyDescriptor description */
         Atomic<unsigned int> _totalValues; /**< Total number of values */
         Lock                 _lock;        /**< _valueMap exclusive lock */
         ValueMap             _valueMap;    /**< Registered Value elements */
      private:
         /*! \brief InstrumentationKeyDescriptor default constructor (private)
          */
         InstrumentationKeyDescriptor ();
         /*! \brief InstrumentationKeyDescriptor copy constructor (private)
          */
         InstrumentationKeyDescriptor ( InstrumentationKeyDescriptor &ikd );
         /*! \brief InstrumentationKeyDescriptor copy assignment operator (private)
          */
         InstrumentationKeyDescriptor& operator= ( InstrumentationKeyDescriptor &ikd );
      public:
         /*! \brief InstrumentationKeyDescriptor constructor
          */
         InstrumentationKeyDescriptor ( nanos_event_key_t id, const std::string &description, nanos_event_level_t level, bool stacked ) : _id( id ), _level(level),_stacked(stacked), _description ( description ),
                                     _totalValues(1), _lock(), _valueMap() {}

         /*! \brief InstrumentationKeyDescriptor constructor
          */
         InstrumentationKeyDescriptor ( nanos_event_key_t id, const char *description, nanos_event_level_t level, bool stacked ) : _id( id ), _level(level), _stacked(stacked),  _description ( description ),
                                     _totalValues(1), _lock(), _valueMap() {}

         /*! \brief InstrumentationKeyDescriptor destructor
          */
         ~InstrumentationKeyDescriptor() {}

         /*! \brief Gets key descriptor id
          */
         nanos_event_key_t getId ( void ) const;

         /*! \brief Set the event level
          */
         void setLevel ( nanos_event_level_t value );

         /*! \brief Normalize level
          */
         void normalizeLevel ( nanos_event_level_t value );

         /*! \brief return if the event is stacked
          */
         bool isStacked ( void );

         /*! \brief Gets key descriptor textual description
          */
         const std::string getDescription ( void ) const;

         /*! \brief Inserts (or gets) a value into (from) valueMap
          */
         nanos_event_value_t registerValue ( const std::string &value, const std::string &description="", bool abort_when_registered=true );

         /*! \brief Inserts (or gets) a value into (from) valueMap
          */
         nanos_event_value_t registerValue ( const char *value, const char *description="", bool abort_when_registered=true );

         /*! \brief Inserts a value into valueMap (the value is given by user)
          */
         void registerValue ( const std::string &value, nanos_event_value_t val,
                              const std::string &description="", bool abort_when_registered=true );

         /*! \brief Inserts a value into valueMap (the value is given by user)
          */
         void registerValue ( const char *value, nanos_event_value_t val,
                              const char *description="", bool abort_when_registered=true );
         /*! \brief Gets a value into (from) valueMap
          */
         nanos_event_value_t getValue ( const std::string &value );

         /*! \brief Gets a value into (from) valueMap
          */
         nanos_event_value_t getValue ( const char *value );

         /*! \brief Returns starting point of valueMap ( iteration purposes )
          */
         ConstValueMapIterator beginValueMap ( void );

         /*! \brief Returns ending point of valueMap ( iteration purposes )
          */
         ConstValueMapIterator endValueMap ( void );

         /*! \brief Returns a Value description for a given value
          */
         const std::string getValueDescription ( nanos_event_value_t val );

         /*! \brief Returns the number of registered values
          */
         size_t getSize( void ) const;
   };

/*!\class InstrumentationDictionary
 * \brief InstrumentationDictionary is event's dictionary.
 * \description It allows to register and recover keys and pairs of <key,value> objects given them an internal code which can be used as identifier. The dictionary also allow to associate a description to each key and <key,value> objects.
 *
 */ 
   class InstrumentationDictionary {
      public:
         typedef TR1::unordered_map<std::string, InstrumentationKeyDescriptor*> KeyMap;
         typedef KeyMap::iterator KeyMapIterator;
         typedef KeyMap::const_iterator ConstKeyMapIterator;
      private:
         Atomic<unsigned int> _totalKeys; /**< Total number of keys */
         Lock                 _lock;      /**< Is the _keyMap exclusive lock */
         KeyMap               _keyMap;    /**< Registered Key elements */
         nanos_event_level_t  _level;     //!< Default verbosity level

         /*! \brief InstrumentationDictionary copy constructor (private)
          */
         InstrumentationDictionary ( InstrumentationDictionary &id );
         /*! \brief InstrumentationDictionary copy assignment operator (private)
          */
         InstrumentationDictionary& operator= ( InstrumentationDictionary &id );
      public:
         /*! \brief InstrumentationDictionary default constructor
          */
         InstrumentationDictionary () : _totalKeys(1), _lock(), _keyMap(), _level(EVENT_DEFAULT)
         {
            /* ******************************************** */
            /* Instrumentation events: In order initialization */
            /* ******************************************** */

            /* 01 */ registerEventKey("api","Nanos Runtime API", true, EVENT_DEVELOPER, true );
            registerEventValue("api","find_slicer","nanos_find_slicer()");
            registerEventValue("api","wg_wait_completion","nanos_wg_wait_completion()");
            registerEventValue("api","*_create_sync_cond","nanos_create_xxx_cond()");
            registerEventValue("api","sync_cond_wait","nanos_sync_cond_wait()");
            registerEventValue("api","sync_cond_signal","nanos_sync_cond_signal()");
            registerEventValue("api","destroy_sync_cond","nanos_destroy_sync_cond()");
            registerEventValue("api","wait_on","nanos_wait_on()");
            registerEventValue("api","init_lock","nanos_init_lock()");
            registerEventValue("api","set_lock","nanos_set_lock()");
            registerEventValue("api","unset_lock","nanos_unset_lock()");
            registerEventValue("api","try_lock","nanos_try_lock()");
            registerEventValue("api","destroy_lock","nanos_destroy_lock()");
            registerEventValue("api","single_guard","nanos_single_guard()");
            registerEventValue("api","team_barrier","nanos_team_barrier()");
            registerEventValue("api","current_wd", "nanos_current_wd()");
            registerEventValue("api","get_wd_id","nanos_get_wd_id()");
            registerEventValue("api","*_create_wd","nanos_create_xxx_wd()");
            registerEventValue("api","submit","nanos_submit()");
            registerEventValue("api","create_wd_and_run","nanos_create_wd_and_run()");
            registerEventValue("api","set_internal_wd_data","nanos_set_internal_wd_data()");
            registerEventValue("api","get_internal_wd_data","nanos_get_internal_wd_data()");
            registerEventValue("api","yield","nanos_yield()");
            registerEventValue("api","create_team","nanos_create_team()");
            registerEventValue("api","enter_team","nanos_enter_team()");
            registerEventValue("api","leave_team","nanos_leave_team()");
            registerEventValue("api","end_team","nanos_end_team()");
            registerEventValue("api","get_addr","nanos_get_addr()");
            registerEventValue("api","copy_value","nanos_copy_value()");
            registerEventValue("api","omp_barrier","nanos_omp_barrier()");
            registerEventValue("api","get_starring_threads","nanos_get_implicit_threads()");
            registerEventValue("api","get_supporting_threads","nanos_get_nonimplicit_threads()");
            registerEventValue("api","omp_find_worksharing","nanos_omp_find_worksharing()");
            registerEventValue("api","omp_get_schedule","nanos_omp_get_schedule()");
            registerEventValue("api","malloc","nanos_malloc()");
            registerEventValue("api","free","nanos_free()");
            registerEventValue("api","get_num_ready_tasks","nanos_get_num_ready_tasks()");
            registerEventValue("api","get_num_total_tasks","nanos_get_num_total_tasks()");
            registerEventValue("api","get_num_nonready_tasks","nanos_get_num_nonready_tasks()");
            registerEventValue("api","get_num_blocked_tasks","nanos_get_num_blocked_tasks()");
            registerEventValue("api","get_num_running_tasks","nanos_get_num_running_tasks()");
            registerEventValue("api","dependence_pendant_writes","nanos_dependence_pendant_writes()");
            registerEventValue("api","in_final","nanos_in_final()");
            registerEventValue("api","set_final","nanos_set_final()");
            registerEventValue("api","dependence_release_all","nanos_dependence_release_all()");
            registerEventValue("api","set_translate_function","nanos_set_translate_function()");
            registerEventValue("api","memalign","nanos_memalign()");
            registerEventValue("api","cmalloc","nanos_cmalloc()");
            registerEventValue("api","stick_to_producer","nanos_stick_to_producer()");
            registerEventValue("api","task_reduction_register","nanos_task_reduction_register()");
            registerEventValue("api","task_reduction_get_thread_storage","nanos_task_reduction_get_thread_storage()");

            /* 02 */ registerEventKey("wd-id","Work Descriptor id:", true, EVENT_DEVELOPER, true);

            /* 03 */ registerEventKey("cache-copy-in","Transfer data into device cache", true, EVENT_DISABLED );
            /* 04 */ registerEventKey("cache-copy-out","Transfer data to main memory", true, EVENT_DISABLED );
            /* 05 */ registerEventKey("cache-local-copy","Local copy in device memory", true, EVENT_DISABLED );
            /* 06 */ registerEventKey("cache-malloc","Memory allocation in device cache", true, EVENT_DISABLED );
            /* 07 */ registerEventKey("cache-free","Memory free in device cache", true, EVENT_DISABLED );
            /* 08 */ registerEventKey("cache-hit","Hit in the cache", true, EVENT_DISABLED );

            /* 09 */ registerEventKey("copy-in","Copying WD inputs", true, EVENT_DISABLED );
            /* 10 */ registerEventKey("copy-out","Copying WD outputs", true, EVENT_DISABLED );

            /* 11 */ registerEventKey("user-funct-name","User Function Name", true, EVENT_USER, true);

            /* 12 */ registerEventKey("user-code","User Code (wd)", true, EVENT_DISABLED );

            /* 13 */ registerEventKey("create-wd-id","Create WD Id:", true, EVENT_ADVANCED ); // TODO: consider to disable
            /* 14 */ registerEventKey("create-wd-ptr","Create WD pointer:", true, EVENT_DEVELOPER );
            /* 15 */ registerEventKey("wd-num-deps","Create WD num. deps.", true, EVENT_ADVANCED );
            /* 16 */ registerEventKey("wd-deps-ptr","Create WD dependence pointer", true, EVENT_ADVANCED );

            /* 17 */ registerEventKey("lock-addr","Lock address", true, EVENT_DEVELOPER );

            /* 18 */ registerEventKey("num-spins","Number of Spins", true, EVENT_DEVELOPER );
            /* 19 */ registerEventKey("num-yields","Number of Yields", true, EVENT_DEVELOPER );
            /* 20 */ registerEventKey("time-yields","Time on Yield (in nsecs)", true, EVENT_DEVELOPER );

            /* 21 */ registerEventKey("user-funct-location","User Function Location", true, EVENT_USER, true);

            /* 22 */ registerEventKey("num-ready","Number of ready tasks in the queues", true, EVENT_USER );
            /* 23 */ registerEventKey("graph-size","Number tasks in the graph", true, EVENT_USER );

            /* 24 */ registerEventKey("loop-lower","Loop lower bound", true, EVENT_DEVELOPER );
            /* 25 */ registerEventKey("loop-upper","Loop upper", true, EVENT_DEVELOPER );
            /* 26 */ registerEventKey("loop-step","Loop step", true, EVENT_DEVELOPER );

            /* 27 */ registerEventKey("in-cuda-runtime","Inside CUDA runtime", true, EVENT_DEVELOPER );
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_MALLOC_EVENT", "cudaMalloc()" );                                     /* 1 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_FREE_EVENT", "cudaFree()" );                                         /* 2 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_MALLOC_HOST_EVENT", "cudaMallocHost()" );                            /* 3 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_FREE_HOST_EVENT", "cudaFreeHost()" );                                /* 4 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_MEMCOPY_EVENT", "cudaMemcpyDeviceToDevice()" );                      /* 5 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_MEMCOPY_TO_HOST_EVENT", "cudaMemcpyDeviceToHost()" );                /* 6 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_MEMCOPY_TO_DEVICE_EVENT", "cudaMemcpyHostToDevice()" );              /* 7 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_MEMCOPY_ASYNC_EVENT", "cudaMemcpyPeerAsync()" );                     /* 8 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_MEMCOPY_ASYNC_TO_HOST_EVENT", "cudaMemcpyAsyncDeviceToHost()" );     /* 9 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_MEMCOPY_ASYNC_TO_DEVICE_EVENT", "cudaMemcpyAsyncHostToDevice()" );   /* 10 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_INPUT_STREAM_SYNC_EVENT", "cudaInputStreamSynchronize()" );          /* 11 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_OUTPUT_STREAM_SYNC_EVENT", "cudaOutputStreamSynchronize()" );        /* 12 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_KERNEL_STREAM_SYNC_EVENT", "cudaKernelStreamSynchronize()" );        /* 13 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_DEVICE_SYNC_EVENT", "cudaDeviceSynchronize()" );                     /* 14 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_SET_DEVICE_EVENT", "cudaSetDevice()" );                              /* 15 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_GET_DEVICE_PROPS_EVENT", "cudaGetDeviceProperties()" );              /* 16 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_SET_DEVICE_FLAGS_EVENT", "cudaSetDeviceFlags()" );                   /* 17 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_GET_LAST_ERROR_EVENT", "cudaGetLastError()" );                       /* 18 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_GENERIC_EVENT", "CUDA generic event" );                              /* 19 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_MEMCOPY_EVENT", "memcpy()" );                                             /* 20 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_EVENT_CREATE_EVENT", "cudaEventCreate()" );                          /* 21 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_EVENT_DESTROY_EVENT", "cudaEventDestroy()" );                        /* 22 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_EVENT_RECORD_EVENT", "cudaEventRecord()" );                          /* 23 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_EVENT_QUERY_EVENT", "cudaEventQuery()" );                            /* 24 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_EVENT_SYNC_EVENT", "cudaEventSynchronize()" );                       /* 25 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_KERNEL_LAUNCH_EVENT", "Launching CUDA kernel(s) in task" );          /* 26 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_STREAM_CREATE_EVENT", "cudaStreamCreate()" );                        /* 27 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_STREAM_DESTROY_EVENT", "cudaStreamDestroy" );                        /* 28 */
            registerEventValue("in-cuda-runtime", "NANOS_GPU_CUDA_GET_PCI_BUS_EVENT", "cudaDeviceGetPCIBusId()" );                     /* 29 */


            /* 28 */ registerEventKey("xfer-size","Transfer size", true, EVENT_DEVELOPER );

            /* 29 */ registerEventKey("cache-wait","Cache waiting for something", true, EVENT_DISABLED );
            registerEventValue("cache-wait","registerCacheAccess() L.94","registerCacheAccess() waiting for data allocation (not registered in directory)");
            registerEventValue("cache-wait","registerCacheAccess() L.112","registerCacheAccess() waiting for data invalidation in another cache (new entry)");
            registerEventValue("cache-wait","registerCacheAccess() L.122","registerCacheAccess() waiting for data to have no owner");
            registerEventValue("cache-wait","registerCacheAccess() L.141","registerCacheAccess() waiting for data allocation (registered in directory)");
            registerEventValue("cache-wait","registerCacheAccess() L.163","registerCacheAccess() waiting for data invalidation (size has changed)");
            registerEventValue("cache-wait","registerCacheAccess() L.185","registerCacheAccess() waiting for data invalidation in another cache (size has changed)");
            registerEventValue("cache-wait","registerCacheAccess() L.221","registerCacheAccess() waiting for data to be copied back (size has changed)");
            registerEventValue("cache-wait","registerCacheAccess() L.239","registerCacheAccess() waiting for data invalidation in another cache (old version)");
            registerEventValue("cache-wait","registerCacheAccess() L.260","registerCacheAccess() invalidating another cache");
            registerEventValue("cache-wait","registerCacheAccess() L.292","registerCacheAccess() waiting for resize");
            registerEventValue("cache-wait","registerCacheAccess() L.300","registerCacheAccess() waiting for flush");
            registerEventValue("cache-wait","freeSpaceToFit()","freeSpaceToFit()");
            registerEventValue("cache-wait","waitInput()","waitInput()");

            /* 30 */ registerEventKey("chunk-size","Chunk size", true, EVENT_DEVELOPER );

            /* 31 */ registerEventKey("num-blocks","Number of blocking operations", true, EVENT_DEVELOPER );
            /* 32 */ registerEventKey("time-blocks","Time on block (in nsecs)", true, EVENT_DEVELOPER );

            /* 33 */ registerEventKey("num-scheds","Number of scheduler operations", true, EVENT_DEVELOPER );
            /* 34 */ registerEventKey("time-scheds","Time on scheduler operations (in nsecs)", true, EVENT_DEVELOPER );

            /* 35 */ registerEventKey("sched-versioning","Versioning scheduler decisions", true, EVENT_ADVANCED );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_SETDEVICE_CANRUN", "Set WD device + thread can run" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_SETDEVICE_CANNOTRUN", "Set WD device + thread cannot run" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_SELECTWD_FIRSTCANRUN", "Select WD first record + thread can run" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_SELECTWD_FIRSTCANNOTRUN", "Select WD first record + thread cannot run" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_SELECTWD_BELOWMINRECCANRUN", "Select WD not enough records + thread can run" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_SELECTWD_UNDEFINED", "Select WD undefined" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_SELECTWD_GETFIRST", "Select WD get first queue task" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_ATIDLE_GETFIRST", "At idle get first queue task" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_ATIDLE_NOFIRST", "At idle no first queue task found" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_ATPREFETCH_GETFIRST", "At prefetch get first queue task" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_ATPREFETCH_GETIMMSUCC", "At prefetch get immediate successor" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_ATPREFETCH_NOFIRST", "At prefetch no first queue task found" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_ATBEFEX_GETFIRST", "At before exit get first queue task" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_ATBEFEX_NOFIRST", "At before exit no first queue task found" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_SETEARLIESTEW_FOUND", "Set earliest execution worker found" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_SETEARLIESTEW_NOTFOUND", "Set earliest execution worker not found" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_FINDEARLIESTEW_BETTERTIME", "Found earliest execution worker timing reason" );
            registerEventValue("sched-versioning", "NANOS_SCHED_VER_FINDEARLIESTEW_IDLEWORKER", "Found earliest execution worker idle reason" );

            /* 36 */ registerEventKey("dependence","Dependence analysis", true, EVENT_DEVELOPER ); /* System have found a new dependence */
            /* 37 */ registerEventKey("dep-direction", "Dependence direction", true, EVENT_DEVELOPER );

            /* 38 */ registerEventKey("wd-priority","Priority of a work descriptor", true, EVENT_DEVELOPER );

            /* 39 */ registerEventKey("in-opencl-runtime","Inside OpenCL runtime", true, EVENT_DEVELOPER );
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_ALLOC_EVENT", "clCreateBuffer()" );                                     /* 1 */
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_FREE_EVENT", "clReleaseMemObject()" );                                         /* 2 */
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_GET_DEV_INFO_EVENT", "clGetDeviceInfo()" );                            /* 3 */
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_CREATE_CONTEXT_EVENT", "clCreateContext()" );                                /* 4 */
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_MEMWRITE_SYNC_EVENT", "clEnqueueWriteBuffer(blocking=true)" );                      /* 5 */
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_MEMREAD_SYNC_EVENT", "clEnqueueReadBuffer(blocking=true)" );                /* 6 */
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_CREATE_COMMAND_QUEUE_EVENT", "clCreateCommandQueue()" );                /* 7 */
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_GET_PROGRAM_EVENT", "Compile, build and clCreateKernel() nanox routine" );                /* 8 */
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_COPY_BUFFER_EVENT", "clEnqueueCopyBuffer() Device to device transfer" );                /* 9 */
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_CREATE_SUBBUFFER_EVENT", "clCreateSubBuffer(blocking=true)" );                /* 10 */
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_MAP_BUFFER_SYNC_EVENT", "clEnqueueMapBuffer(blocking=true)" );                /* 11 */
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_UNMAP_BUFFER_SYNC_EVENT", "clEnqueueUnmapMemObject(blocking=true)" );                /* 12 */
            registerEventValue("in-opencl-runtime", "NANOS_OPENCL_GENERIC_EVENT", "OpenCL generic event" );                              /* 13 */

            /* 40 */ registerEventKey("taskwait", "Call to the taskwait nanos runtime function", true, EVENT_USER );
            /* 41 */ registerEventKey("set-num-threads","Number of Threads", true, EVENT_USER );
            /* 42 */ registerEventKey("cpuid","Thread cpuid", true, EVENT_USER );

            /* 43 */ registerEventKey("dep-address", "Dependence address", true, EVENT_DEVELOPER );
            /* 44 */ registerEventKey("copy-data-in","WD id that is copying data in", true, EVENT_DEVELOPER );
            /* 45 */ registerEventKey("cache-copy-data-in","WD id that is copying data in", true, EVENT_DEVELOPER );
            /* 46 */ registerEventKey("cache-copy-data-out","WD id that is copying data out", true, EVENT_DEVELOPER );
            /* 47 */ registerEventKey("sched-affinity-constraint","Constraint used by affinity scheduler", true, EVENT_DEVELOPER );

            /* 48*/ registerEventKey("in-mpi-runtime","Inside MPI runtime", true, EVENT_DEVELOPER );
            registerEventValue("in-mpi-runtime", "NANOS_MPI_ALLOC_EVENT", "malloc()" );                                     /* 1 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_FREE_EVENT", "free()" );                                         /* 2 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_DEEP_BOOSTER_ALLOC_EVENT", "deep_booster_alloc(...)" );                            /* 3 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_COPYIN_SYNC_EVENT", "MPI Copy data to remote node" );                                /* 4 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_COPYOUT_SYNC_EVENT", "MPI copy data from remote node" );                      /* 5 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_COPYDEV2DEV_SYNC_EVENT", "MPI copy data between two nodes" );                      /* 6 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_DEEP_BOOSTER_FREE_EVENT", "deep_booster_free(...)" );                      /* 7 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_INIT_EVENT", "nanos_mpi_init()" );                      /* 8 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_FINALIZE_EVENT", "nanos_mpi_finalize()" );                      /* 9 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_SEND_EVENT", "nanos mpi_send()" );                      /* 10 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_RECV_EVENT", "nanos mpi_recv()" );                      /* 11 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_SSEND_EVENT", "nanos mpi_ssend()" );                      /* 12 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_COPYLOCAL_SYNC_EVENT", "MPI Copy local inside node" );                      /* 13 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_REALLOC_EVENT", "MPI Realloc data inside node" );                      /* 14 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_WAIT_FOR_COPIES_EVENT", "MPI remote node task waiting for copyIns before starting" );   /* 15 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_RNODE_COPYIN_EVENT", "MPI remote node cache doing copy in" );  /* 16 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_RNODE_COPYOUT_EVENT", "MPI remote node cache doing copy out" );            /* 17 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_RNODE_DEV2DEV_IN_EVENT", "MPI remote node cache doing device to device copy in" );   /* 18 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_RNODE_DEV2DEV_OUT_EVENT", "MPI remote node cache doing device to device copy outg" );  /* 19 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_RNODE_ALLOC_EVENT", "MPI remote node cache doing allocation" );   /* 20 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_RNODE_REALLOC_EVENT", "MPI remote node cache doing reallocation" );  /* 21 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_RNODE_FREE_EVENT", "MPI remote node cache doing free of previous allocated chunk" ); /* 22 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_RNODE_COPYLOCAL_EVENT", "MPI remote node cache doing copy local" );  /* 23 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_IRECV_EVENT", "Async recv" ); /* 24 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_ISEND_EVENT", "Async send" );  /* 25 */
            registerEventValue("in-mpi-runtime", "NANOS_MPI_GENERIC_EVENT", "MPI generic event" );                /* 26 */

            /* 49 */ registerEventKey("wd-ready", "Work descriptor becomes ready", false, EVENT_ADVANCED );
            /* 50 */ registerEventKey("wd-blocked", "Work descriptor becomes blocked", false, EVENT_ADVANCED );
            /* 51 */ registerEventKey("parallel-outline-fct", "Parallel Outline Function", false, EVENT_ADVANCED );

            /* 52 */ registerEventKey("async-thread","Asynchronous thread state events", true, EVENT_DEVELOPER );
            registerEventValue("async-thread", "ASYNC_THREAD_INLINE_WORK_DEP_EVENT", "inlineWorkDependent()" );  /* 1 */
            registerEventValue("async-thread", "ASYNC_THREAD_PRE_RUN_EVENT", "WD pre-run" );                     /* 2 */
            registerEventValue("async-thread", "ASYNC_THREAD_RUN_EVENT", "Running WD" );                         /* 3 */
            registerEventValue("async-thread", "ASYNC_THREAD_POST_RUN_EVENT", "WD post-run" );                   /* 4 */
            registerEventValue("async-thread", "ASYNC_THREAD_SCHEDULE_EVENT", "Scheduling tasks" );              /* 5 */
            //registerEventValue("async-thread", "ASYNC_THREAD_WAIT_INPUTS_EVENT", "Waiting for inputs" );         /* 5 */
            registerEventValue("async-thread", "ASYNC_THREAD_CHECK_WD_INPUTS_EVENT", "Checking for inputs" );    /* 6 */
            registerEventValue("async-thread", "ASYNC_THREAD_CHECK_WD_OUTPUTS_EVENT", "Checking for outputs" );  /* 7 */
            registerEventValue("async-thread", "ASYNC_THREAD_CP_DATA_IN_EVENT", "Copy data in" );                /* 8 */
            registerEventValue("async-thread", "ASYNC_THREAD_CP_DATA_OUT_EVENT", "Copy data out" );              /* 9 */
            registerEventValue("async-thread", "ASYNC_THREAD_CHECK_EVTS_EVENT", "Check events" );                /* 10 */
            registerEventValue("async-thread", "ASYNC_THREAD_PROCESS_EVT_EVENT", "Processing finished event" );  /* 11 */
            registerEventValue("async-thread", "ASYNC_THREAD_SYNCHRONIZE_EVENT", "Synchronize copy" );           /* 12 */

            /* 53 */ registerEventKey("copy-in-gpu", "Asynchronous memory copy from host to device", true, EVENT_DISABLED );
            /* 54 */ registerEventKey("copy-out-gpu", "Asynchronous memory copy from device to host", true, EVENT_DISABLED );
            /* 55 */ registerEventKey("gpu-wd-id","GPU Work Descriptor id:", true, EVENT_DEVELOPER, true );

            /* 56 */ registerEventKey("wd-criticality","Work descriptor criticality", true,  EVENT_DEVELOPER );
            /* 57 */ registerEventKey("blev-overheads", "Total overheads of botlev scheduler", true, EVENT_DEVELOPER );
            /* 58 */ registerEventKey("blev-overheads-breakdown", "Overheads of botlev scheduler broken down", true,  EVENT_DEVELOPER );
            /* 59 */ registerEventKey("critical-wd-id", "A critical work descriptor is submitted", true, EVENT_DEVELOPER );

            /* 60 */ registerEventKey("copy-dir-devices", "Asynchronous memory copy between host and devices", true , EVENT_DEVELOPER );
            registerEventValue("copy-dir-devices", "NANOS_DEVS_CPDIR_H2D_GPU_EVENT", "Host to GPU device transfer (CUDA)" );                     /* 1 */
            registerEventValue("copy-dir-devices", "NANOS_DEVS_CPDIR_D2H_GPU_EVENT", "GPU device to host transfer (CUDA)" );                     /* 2 */
            /* 61 */ registerEventKey("concurrent-tasks", "Number of concurrent tasks in the ready queue", false, EVENT_DISABLED );
            /* 62 */ registerEventKey("network-transfer", "Network transfer to node ", false, EVENT_DEVELOPER);
            /* 63 */ registerEventKey("team-ptr", "Team info", false, EVENT_DEVELOPER);

            /* 64 */ registerEventKey("thread-numa-node","Thread NUMA node", true, EVENT_ADVANCED);
            /* 65 */ registerEventKey("wd-numa-node","WD NUMA node", true, EVENT_ADVANCED);

            /* 66 */ registerEventKey("steal","Stolen WD", true, EVENT_DEVELOPER );

            /* 67 */ registerEventKey("in-xdma", "Inside xdma FPGA DMA library", true, EVENT_ADVANCED);
            registerEventValue("in-xdma", "NANOS_FPGA_OPEN_EVENT", "xdma open()");                       /* 1 */
            registerEventValue("in-xdma", "NANOS_FPGA_CLOSE_EVENT", "xdma close");                     /* 2 */
            registerEventValue("in-xdma", "NANOS_FPGA_REQ_CHANNEL_EVENT", "xdma open channel");          /* 3 */
            registerEventValue("in-xdma", "NANOS_FPGA_REL_CHANNEL_EVENT", "xdma close channel");          /* 4 */
            registerEventValue("in-xdma", "NANOS_FPGA_SUBMIT_IN_DMA_EVENT", "xdma submit in");      /* 5 */
            registerEventValue("in-xdma", "NANOS_FPGA_SUBMIT_OUT_DMA_EVENT", "xdma submit out");    /* 5 */
            registerEventValue("in-xdma", "NANOS_FPGA_WAIT_INPUT_DMA_EVENT", "xdma wait in");                /* 6 */
            registerEventValue("in-xdma", "NANOS_FPGA_WAIT_OUTPUT_DMA_EVENT", "xdma wait out");                /* 7 */
            /* 68 */ registerEventKey("accelerator#", "Accelerator on which task is being executed", true, EVENT_ADVANCED);

            /* 69 */ registerEventKey("reduction", "Reduction support", true, EVENT_DEVELOPER);
            registerEventValue("reduction", "RED_REQUEST_NEW_STORAGE", "Allocating private storage" ); /* 1 */
            registerEventValue("reduction", "RED_COMMIT_ALL", "Reducing private storages" );           /* 2 */
            /* 70 */ registerEventKey("network-transfer", "Network transfer to node ", false, EVENT_ADVANCED);
            /* 71 */ registerEventKey("cache-evict", "Cache eviction", false, EVENT_ADVANCED);
            /* 72 */ registerEventKey("copy-data-alloc","Cache allocation", false, EVENT_ADVANCED);

            /* ** */ registerEventKey("debug","Debug Key", true, EVENT_ADVANCED ); /* Keep this key as the last one */
         }

         /*! \brief InstrumentationDictionary destructor
          */
         ~InstrumentationDictionary() {}

         /*! \brief Normalize levels
          */
         void normalizeLevels ( void );

         /*! \brief Inserts (or gets) a key into (from) the keyMap
          */
         nanos_event_key_t registerEventKey ( const std::string &key, const std::string &description="", bool abort_when_registered=true, nanos_event_level_t level=EVENT_ENABLED, bool stacked=false );

         /*! \brief Inserts (or gets) a key into (from) the keyMap
          */
         nanos_event_key_t registerEventKey ( const char *key, const char *description="", bool abort_when_registered=true, nanos_event_level_t level=EVENT_ENABLED, bool stacked=false );

         /*! \brief Gets a key into (from) the keyMap
          */
         nanos_event_key_t getEventKey ( const std::string &key );

         /*! \brief Gets a key into (from) the keyMap
          */
         nanos_event_key_t getEventKey ( const char *key );

         /*! \brief Inserts (or gets) a value into (from) the valueMap (which belongs to 'key' parameter )
          */
         nanos_event_value_t registerEventValue ( const std::string &key, const std::string &value,
                                                  const std::string &description="", bool abort_when_registered=true );

         /*! \brief Inserts (or gets) a value into (from) the valueMap (which belongs to 'key' parameter )
          */
         nanos_event_value_t registerEventValue ( const char *key, const char *value, const char *description="", bool abort_when_registered=true );

         /*! \brief Inserts a value into the valueMap, which belongs to 'key' parameter (value is given by user)
          */
         void registerEventValue ( const std::string &key, const std::string &value,
                                   nanos_event_value_t val,
                                   const std::string &description="", bool abort_when_registered=true );

         /*! \brief Inserts a value into the valueMap, which belongs to 'key' parameter (value is given by user)
          */
         void registerEventValue ( const char *key, const char *value, nanos_event_value_t val,
                                   const char *description="", bool abort_when_registered=true );

         /*! \brief Enable/disable all events in dictionary
          */
         void setDefaultLevel ( nanos_event_level_t level );

         /*! \brief Get the list of enabled and disabled events
          */
         std::string getSummary();

         /*! \brief Enable/disable all events prefixed with prefix
          */
         void switchEventPrefix ( const char *prefix, nanos_event_level_t level );

         /*! \brief Gets a value into (from) the valueMap (which belongs to 'key' parameter )
          */
         nanos_event_value_t getEventValue ( const std::string &key, const std::string &value );

         /*! \brief Gets a value into (from) the valueMap (which belongs to 'key' parameter )
          */
         nanos_event_value_t getEventValue ( const char *key, const char *value );

         /*! \brief Returns starting point of keyMap ( iteration purposes )
          */
         ConstKeyMapIterator beginKeyMap ( void );

         /*! \brief Returns ending point of keyMap ( iteration purposes )
          */
         ConstKeyMapIterator endKeyMap ( void );

         /*! \brief Returns a Key description for a given key
          */
         const std::string getKeyDescription ( nanos_event_key_t key );

         /*! \brief Returns a Value description for a given key and a value
          */
         const std::string getValueDescription ( nanos_event_key_t key, nanos_event_value_t val );

   };
#endif

//! \class Instrumentation
//! \brief Instrumentation main class is the core of the insrumentation behaviour.
/*! \description This class implements several methods: methods to create events, methods to raise event, WorkDescriptor context swhitch methods. Instrumentation plugins will be derived classes of this class. These plugins must implement (at least) a subset of these methods will determine their specific behaviour. This is called the instrumentation plugin interface:
 *
 * \subsubsection instrumentation_api Instrumentation API
 *
 *  - initialize(): this method is executed at runtime startup and can be used to create buffers, auxiliary structures, initialize values (e.g. time stamp), etc.
 *  - finalize(): this method is executed at runtime shutdown and can be used to dump remaining data into a file or standard output, post-process trace information, delete buffers and auxiliary structures, etc.
 *  - addEventList(): this method is executed each time the runtime raises an event. It receives a list of events (EventList) and the specific instrumentation class has to deal with each event in this list in order to generate (or not) a valid output.
 *  - disable(): this method disables instrumentation until enable method is call again. Disable method will inform plugin to stop instrumenting but runtime's core will continue generating events, so it is suposed that the plugin will ignore or keep them to stay in a consistent state.
 *  - enable(): this method enables again the instrumentation, previously disabled by disable method.
 *  - threadStart(): this method is executed each time a new thread starts.
 *  - threadFinish(): this method is executed each time a new thread finishes.
 *  - addResumeTask(): this method is executed each time a task is resumed in the current thread
 *  - addSuspendTask(): this method is executed each time a task is suspended in the current thread
 *  - incrementMaxThreads() this method is executed each time we increase the number of threads
 *
 *  The Instrumentation object implementation is based in the concept of plugins which allow that several implementations based on its interface can be used without having to modify the runtime library. As we can see in the class diagram we have a generic class which defines all the instrumentation interface and several specific classes which defines the specific output format. But specific Instrumentation programmers can also overload other base methods in order to get an specific behavior when the plugin is invoked. Derived classes have to define (at least) the previously mentioned virtual methods:
 *
 *  \code
 *  void initialize( void );
 *  void finalize( void );
 *  void addEventList(  unsigned int count, Event *events );
 *  void enable( void );
 *  void disable ( void );
 *  void threadStart ( BaseThread &thread ) = 0;
 *  void threadFinish ( BaseThread &thread ) = 0;
 *  void addResumeTask( WorkDescriptor &w ) = 0 ;
 *  void addSuspendTask( WorkDescriptor &w, bool last = false ) = 0 ;
 *  void incrementMaxThreads( void ) {}
 *  \endcode
 *
 *  Instrumentation also specify as virtual functions some generic services which can be used at runtime code. These services are grouped in:
 *
 *  - Create event's services: these services are focused in create specific event objects. Usually they are not called by external agents but they are used by raise event's services (explained below).
 *  - Raise event's services: these services are focused in effectively producing an event (or list of events) which will be visible by the user. Usually these functions will call one or several create event's service(s) and finally produce an effective output by calling plugin's addEventList() service.
 *  - Context switch's services: they are used to backup/restore the instrumentation information history for the current WorkDescriptor? (see InstrumentationContext class).
 *
 *  Finally, Instrumentation class also offers two more services to enable/disable state instrumentation. Once the user calls disableStateInstrumentation() the runtime will not produce more state events until the user enable it by calling enableStateInstrumentation(). Although no state events will be produced during this interval of time Instrumentation class will keep all potential state changes by creating a special event object: the substate event.
 *
*/
   class Instrumentation
   {
      public:
         class Event {
            private:
               nanos_event_type_t          _type;         /**< Event type */
               nanos_event_key_t           _key;          /**< Event key */
               nanos_event_value_t         _value;        /**< Event value */
               nanos_event_domain_t        _ptpDomain;    /**< A specific domain in which ptpId is unique */
               nanos_event_id_t            _ptpId;        /**< PtP event id */
               unsigned int                _partner;      /**< PtP communication partner (destination or origin), only applies to Cluster (is always 0 in smp) */


            public:
               /*! \brief Event default constructor
                *
                *  \see State Burst Point PtP
                */
               Event () : _type((nanos_event_type_t) 0), _key(0), _value(0),
                          _ptpDomain((nanos_event_domain_t) 0), _ptpId(0), _partner( NANOX_INSTRUMENTATION_PARTNER_MYSELF ) {}
               /*! \brief Event constructor
                *
                *  Generic constructor used by all other specific constructors
                *
                *  \see State Burst Point PtP
                */
               Event ( nanos_event_type_t type, nanos_event_key_t key, nanos_event_value_t value,
                       nanos_event_domain_t ptp_domain, nanos_event_id_t ptp_id, unsigned int partner = NANOX_INSTRUMENTATION_PARTNER_MYSELF ) :
                     _type (type), _key(key), _value (value),
                     _ptpDomain (ptp_domain), _ptpId (ptp_id), _partner(partner)
               { }

               /*! \brief Event copy constructor
                */
               Event ( const Event & evt )
               {
                  _type = evt._type;
                  _key = evt._key;
                  _value = evt._value;
                  _ptpDomain = evt._ptpDomain;
                  _ptpId     = evt._ptpId;
                  _partner   = evt._partner;

               }

               /*! \brief Event copy assignment operator
                */
               void operator= ( const Event & evt )
               {
                  // self-assignment: ok
                  if ( this == &evt ) return;

                  _type = evt._type;
                  _key = evt._key;
                  _value = evt._value;
                  _ptpDomain = evt._ptpDomain;
                  _ptpId     = evt._ptpId;
                  _partner   = evt._partner;

               }

               /*! \brief Event destructor
                */
               ~Event() {}

               /*! \brief Get event type
                */
               nanos_event_type_t getType () const;

               /*! \brief Get event state
                */
               nanos_event_state_value_t getState ();

               /*! \brief Get key
                */
               nanos_event_key_t getKey () const;

               /*! \brief Get value
                */
               nanos_event_value_t getValue () const;

               /*! \brief Get specific domain ( useful in PtP events)
                *  \see getId
                */
               unsigned int getDomain ( void ) const;

               /*! \brief Get event id (unique in a specific domain, useful in PtP events)
                *  \see getDomain
                */
               long long getId( void ) const;

               /*! \brief Get event partner (destination or origin of a PtP event, only applies to Cluster, returns 0 on SMP)
                *  \see getDomain
                */
               unsigned int getPartner( void ) const;

               /*! \brief Change event type to the complementary value (i.e. if type is BURST_START it changes to BURST_END)
                */
               void reverseType ( );
         };
         class State : public Event {
            private:
              /*! \brief State event default constructor (private)
               */
               State();
              /*! \brief State event copy constructor (private)
               */
               State( State &s);
              /*! \brief State event copy constructor (private)
               */
               State& operator= ( State &s);
            public:
              /*! \brief State event constructor
               */
              State ( nanos_event_type_t type = NANOS_STATE_START, nanos_event_state_value_t state = NANOS_ERROR )
                    : Event (type, 0, (nanos_event_value_t) state, (nanos_event_domain_t) 0, (nanos_event_id_t) 0 ) { }
              friend class Instrumentation;
         };
         class Burst : public Event {
             private:
               /*! \brief Burst event default constructor (private)
                */
               Burst();
               /*! \brief Burst event copy constructor (private)
                */
               Burst( Burst &b);
               /*! \brief Burst event copy constructor (private)
                */
               Burst& operator= ( Burst &b);
             public:
               /*! \brief Burst event constructor
                */
               Burst ( bool start, nanos_event_key_t key, nanos_event_value_t value )
                     : Event ( start? NANOS_BURST_START: NANOS_BURST_END, key, value, (nanos_event_domain_t) 0, (nanos_event_id_t) 0 ) { }
         };
         class Point : public Event {
             private:
               /*! \brief Point event default constructor (private)
                */
               Point();
               /*! \brief Point event copy constructor (private)
                */
               Point( Point &p );
               /*! \brief Point event copy assignment operator (private)
                */
               Point& operator= ( Point &p );
             public:
               /*! \brief Point event constructor
                */
               Point ( nanos_event_key_t key, nanos_event_value_t value )
                     : Event ( NANOS_POINT, key, value, (nanos_event_domain_t) 0, (nanos_event_id_t) 0 ) { }
         };
         class PtP : public Event {
            private:
               /*! \brief PtP event default constructor (private)
                */
               PtP();
               /*! \brief PtP event copy constructor (private)
                */
               PtP( PtP &ptp);
               /*! \brief PtP event copy assignment operator (private)
                */
               PtP& operator= ( PtP &ptp);
            public:
               /*! \brief PtP event constructor
                */
               PtP ( bool start, nanos_event_domain_t domain, nanos_event_id_t id, nanos_event_key_t key,  nanos_event_value_t value, unsigned int partner = NANOX_INSTRUMENTATION_PARTNER_MYSELF );
               friend class Instrumentation;
         };
#ifndef NANOS_INSTRUMENTATION_ENABLED
      public:
         Instrumentation () {}
         ~Instrumentation () {}
#else
      protected: /* They can be accessed by plugins (derived classes ) */
         InstrumentationDictionary      _instrumentationDictionary; /**< Instrumentation Dictionary (allow to register keys and values) */
         InstrumentationContext        &_instrumentationContext; /**< Instrumentation Context */
         bool                           _emitStateEvents;
         bool                           _emitPtPEvents;
         bool                           _emitInternalEvents;
      private:
         /*! \brief Instrumentation default constructor (private)
          */
         Instrumentation();
         /*! \brief Instrumentation copy constructor (private)
          */
         Instrumentation( Instrumentation &i);
         /*! \brief Instrumentation copy assignment operator (private)
          */
         Instrumentation& operator= ( Instrumentation &i);
      public:
         /*! \brief Instrumentation constructor
          */
         Instrumentation( InstrumentationContext &ic ) : _instrumentationDictionary(), _instrumentationContext(ic), _emitStateEvents(true), _emitPtPEvents(true), _emitInternalEvents(false) {}

         /*! \brief Instrumentation destructor
          */
         virtual ~Instrumentation() {}

         /*! \brief Gets InstrumentationDictionary
          *
          */
         InstrumentationDictionary * getInstrumentationDictionary ( void );

         bool isStateEnabled() const;
         bool isPtPEnabled() const;
         bool isInternalsEnabled() const;
         /*! \brief Enable/disable events
          */
         void filterEvents(std::string event_default, std::list<std::string> &enable_events, std::list<std::string> &disable_events );

         // low-level instrumentation interface (pure virtual functions)

         /*! \brief Pure virtual functions executed at the beginning of instrumentation phase
          *
          *  Each of (specific) instrumentation modules have to implement this function in order
          *  to be consistent with the instrumentation model
          */
         virtual void initialize( void ) = 0;

         /*! \brief Pure virtual functions executed at the end of instrumentation phase
          *
          *  Each of (specific) instrumentation modules have to implement this function in order
          *  to be consistent with the instrumentation model
          */
         virtual void finalize( void ) = 0;

         /*! \brief Pure virtual functions executed to enable again instrumentation
          *
          *  Each of (specific) instrumentation modules have to implement this function in order
          *  to be consistent with the instrumentation model
          */
         virtual void enable( void ) = 0;
         /*! \brief Pure virtual functions executed to disable instrumentation
          *
          *  Each of (specific) instrumentation modules have to implement this function in order
          *  to be consistent with the instrumentation model
          */
         virtual void disable( void ) = 0;

         /*! \brief Pure virtual function executed on each thread initialization
          *
          */
         virtual void threadStart( BaseThread &thread ) = 0;

         /*! \brief Pure virtual function executed on each thread finalization
          *
          */
         virtual void threadFinish ( BaseThread &thread ) = 0;

         virtual void addResumeTask( WorkDescriptor &w ) = 0 ;

         virtual void addSuspendTask( WorkDescriptor &w, bool last = false ) = 0 ;

         virtual void incrementMaxThreads( void ) {}

         /*! \brief Pure virtual functions executed each time runtime wants to add an event
          *
          *  Each of (specific) instrumentation modules have to implement this function in order
          *  to be consistent with the instrumentation model. This function includes several
          *  events in a row to facilitate implementation in which several events occurs at
          *  the same time (i.e. same timestamp).
          *
          *  \param[in] count is the number of events
          *  \param[in] events is a vector of 'count' events
          */
         virtual void addEventList ( unsigned int count, Event *events ) = 0;

         // CORE: high-level instrumentation interface (virtual functions)

         /*! \brief Used when creating a work descriptor (initializes instrumentation context associated to a WD)
          */
         virtual void wdCreate( WorkDescriptor* newWD );

         /*! \brief Flush the deferred events (if any) of the given work descriptor
          *
          *  \param[in] wd, this work descriptor's deferred events will be flushed
          */
         virtual void flushDeferredEvents ( WorkDescriptor* wd );

         /*! \brief Used in work descriptor context switch (oldWD has finished completely its execution
          *
          *  \param[in] oldWD, is the work descriptor which leaves the cpu
          *  \param[in] newWD, is the work descriptor which enters the cpu
          */
         virtual void wdSwitch( WorkDescriptor* oldWD, WorkDescriptor* newWD, bool last = false );

         /*! \brief Used by higher levels to create a BURST_START event
          *
          *  \param[in,out] e is an event reference, preallocated by the caller
          *  \param[in] key is the key in the related  pair <key,value>
          *  \param[in] value is the value in related pair <key,value>
          */
         void  createBurstEvent ( Event *e, nanos_event_key_t key, nanos_event_value_t value, InstrumentationContextData *icd = NULL );

         /*! \brief Used by higher levels to create a BURST_END event
          *
          *  \param[in,out] e is an event reference, preallocated by the caller
          *  \param[in] key is the key in the related  pair <key,value>
          *  \param[in] value is the value in related pair <key,value>
          */
         void closeBurstEvent ( Event *e, nanos_event_key_t key, nanos_event_value_t value, InstrumentationContextData *icd = NULL );

         /*! \brief Used by higher levels to create a STATE event
          *
          *  \param[in,out] e is an event reference, preallocated by the caller
          *  \param[in] state is the state value for the event
          */
         void createStateEvent ( Event *e, nanos_event_state_value_t state, InstrumentationContextData *icd = NULL );

         /*! \brief Used by higher levels to create a STATE event (value will be previous state in instrumentation context info)
          *
          *  \param[in,out] e is an event reference, preallocated by the caller
          */
         void returnPreviousStateEvent ( Event *e, InstrumentationContextData *icd = NULL );

         /*! \brief Used by higher levels to create a POINT (punctual) event
          *
          *  The created event will contain a vector of nkvs pairs <key,value> that are build from
          *  separated vectors of keys and values respectively (received as a parameters).
          *
          *  \param[in,out] e is an event reference, preallocated by the caller
          *  \param[in] nkvs is the number of pairs <key,value> related with the new event
          *  \param[in] key is a vector of nkvs keys
          *  \param[in] value is a vector of nkvs  values
          */
         void createPointEvent ( Event *e, nanos_event_key_t key, nanos_event_value_t value );

         /*! \brief Used by higher levels to create a PTP_START event
          *
          *  The created event will contain a vector of nkvs pairs <key,value> that are build from
          *  separated vectors of keys and values respectively (received as a parameters).
          *
          *  \param[in,out] e is an event reference, preallocated by the caller
          *  \param[in] domain specifies a specific domain in which id is a unique value
          *  \param[in] id is a unique id in a given domain context
          *  \param[in] nkvs is the number of pairs <key,value> related with the new event
          *  \param[in] key is a vector of nkvs keys
          *  \param[in] value is a vector of nkvs  values
          *  \param[in] partner is the origin node of the event
          */
         void createPtPStart ( Event *e, nanos_event_domain_t domain, nanos_event_id_t id,
                               nanos_event_key_t keys, nanos_event_value_t values, unsigned int partner = NANOX_INSTRUMENTATION_PARTNER_MYSELF );

         /*! \brief Used by higher levels to create a PTP_END event
          *
          *  The created event will contain a vector of nkvs pairs <key,value> that are build from
          *  separated vectors of keys and values respectively (received as a parameters).
          *
          *  \param[in,out] e is an event reference, preallocated by the caller
          *  \param[in] domain specifies a specific domain in which id is a unique value
          *  \param[in] id is a unique id in a given domain context
          *  \param[in] nkvs is the number of pairs <key,value> related with the new event
          *  \param[in] key is a vector of nkvs keys
          *  \param[in] value is a vector of nkvs  values
          *  \param[in] partner is the destination node of the event
          */
         void createPtPEnd ( Event *e, nanos_event_domain_t domain, nanos_event_id_t id,
                             nanos_event_key_t keys, nanos_event_value_t values, unsigned int partner = NANOX_INSTRUMENTATION_PARTNER_MYSELF );

         /*! \brief Used by higher levels to create a deferred POINT event into a given WorkDescriptor (wd)
          */
         void createDeferredPointEvent ( WorkDescriptor &wd, unsigned int nkvs, nanos_event_key_t *keys,
                                         nanos_event_value_t *values );

         /*! \brief Used by higher levels to create a deferred PTP_START event into a given WorkDescriptor (wd)
          */
         void createDeferredPtPStart ( WorkDescriptor &wd, nanos_event_domain_t domain, nanos_event_id_t id,
                                       nanos_event_key_t key, nanos_event_value_t value, unsigned int partner = NANOX_INSTRUMENTATION_PARTNER_MYSELF );

         /*! \brief Used by higher levels to create a deferred PTP_END event into a given WorkDescriptor (wd)
          */
         void createDeferredPtPEnd ( WorkDescriptor &wd, nanos_event_domain_t domain, nanos_event_id_t id,
                                     nanos_event_key_t key, nanos_event_value_t value, unsigned int partner = NANOX_INSTRUMENTATION_PARTNER_MYSELF );

         void raisePointEvents ( unsigned int nkvs, nanos_event_key_t *key, nanos_event_value_t *val );

         void raiseOpenStateEvent ( nanos_event_state_value_t state );
         void raiseCloseStateEvent ( void );

         void raiseOpenBurstEvent ( nanos_event_key_t key, nanos_event_value_t val );
         void raiseCloseBurstEvent ( nanos_event_key_t key, nanos_event_value_t value );

         void raiseOpenPtPEvent ( nanos_event_domain_t domain, nanos_event_id_t id, nanos_event_key_t key, nanos_event_value_t val, unsigned int partner = NANOX_INSTRUMENTATION_PARTNER_MYSELF );
         void raiseClosePtPEvent ( nanos_event_domain_t domain, nanos_event_id_t id, nanos_event_key_t key, nanos_event_value_t val, unsigned int partner = NANOX_INSTRUMENTATION_PARTNER_MYSELF );

         void raiseOpenStateAndBurst ( nanos_event_state_value_t state, nanos_event_key_t key, nanos_event_value_t val );
         void raiseCloseStateAndBurst ( nanos_event_key_t key, nanos_event_value_t value );
#endif
   };

} // namespace nanos

#endif
