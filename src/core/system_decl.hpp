/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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

#ifndef _NANOS_SYSTEM_DECL_H
#define _NANOS_SYSTEM_DECL_H


#ifdef MPI_DEV
#include "mpi.h"
#endif

#include "processingelement_decl.hpp"
#include "throttle_decl.hpp"
#include <vector>
#include <string>
#include "schedule_decl.hpp"
#include "threadteam_decl.hpp"
#include "slicer_decl.hpp"
#include "worksharing_decl.hpp"
#include "nanos-int.h"
#include "dataaccess_fwd.hpp"
#include "instrumentation_decl.hpp"
#include "directory_decl.hpp"
#include "pminterface_decl.hpp"
#include "cache_map_decl.hpp"
#include "plugin_decl.hpp"
#include "barrier_decl.hpp"

#ifdef GPU_DEV
#include "pinnedallocator_decl.hpp"
#endif

namespace nanos
{

// This class initializes/finalizes the library
// All global variables MUST be declared inside

   class System
   {
      public:
         // constants
         typedef enum { DEDICATED, SHARED } ExecutionMode;
         typedef enum { POOL, ONE_THREAD } InitialMode;
         typedef enum { NONE, WRITE_THROUGH, WRITE_BACK, DEFAULT } CachePolicyType;
         typedef Config::MapVar<CachePolicyType> CachePolicyConfig;

      private:
         // types
         typedef std::vector<PE *>         PEList;
         typedef std::vector<BaseThread *> ThreadList;
         typedef std::map<std::string, Slicer *> Slicers;
         typedef std::map<std::string, WorkSharing *> WorkSharings;
         typedef std::multimap<std::string, std::string> ModulesPlugins;
         
         // globla seeds
         Atomic<int> _atomicWDSeed;

         // configuration variables
         int                  _numPEs;
         int                  _numThreads;
         int                  _deviceStackSize;
         int                  _bindingStart;
         int                  _bindingStride;
         bool                 _bindThreads;
         bool                 _profile;
         bool                 _instrument;
         bool                 _verboseMode;
         ExecutionMode        _executionMode;
         InitialMode          _initialMode;
         bool                 _untieMaster;
         bool                 _delayedStart;
         bool                 _useYield;
         bool                 _synchronizedStart;
         int                  _numSockets;
         int                  _coresPerSocket;

	 // Nanos++ scheduling domain
   	 cpu_set_t            _cpu_set;
   	 int                  _cpu_id[CPU_SETSIZE];
   	 int                  _cpu_count;

         //cutoff policy and related variables
         ThrottlePolicy      *_throttlePolicy;
         SchedulerStats       _schedStats;
         SchedulerConf        _schedConf;

         /*! names of the scheduling, cutoff, barrier and instrumentation plugins */
         std::string          _defSchedule;
         std::string          _defThrottlePolicy;
         std::string          _defBarr;
         std::string          _defInstr;
         /*! Name of the dependencies manager plugin */
         std::string          _defDepsManager;

         std::string          _defArch;

         /*! factories for scheduling, pes and barriers objects */
         peFactory            _hostFactory;
         barrFactory          _defBarrFactory;
         
         /*! Valid plugin map (module)->(list of plugins) */
         ModulesPlugins       _validPlugins;
         

         PEList               _pes;
         ThreadList           _workers;
        
         /*! It counts how many threads have finalized their initialization */
         Atomic<unsigned int> _initializedThreads;
         /*! This counts how many threads we're waiting to be initialized */
         unsigned int         _targetThreads;
         /*! \brief How many threads have been already paused (since the
          scheduler's halt). */
         Atomic<unsigned int> _pausedThreads;
         //! Condition to wait until all threads are paused
         SingleSyncCond<EqualConditionChecker<unsigned int> >  _pausedThreadsCond;
         //! Condition to wait until all threads are un paused
         SingleSyncCond<EqualConditionChecker<unsigned int> >  _unpausedThreadsCond;

         Slicers              _slicers; /**< set of global slicers */

         WorkSharings         _worksharings; /**< set of global worksharings */

         Instrumentation     *_instrumentation; /**< Instrumentation object used in current execution */
         SchedulePolicy      *_defSchedulePolicy;
         
         /*! Dependencies domain manager */
         DependenciesManager *_dependenciesManager;

         /*! It manages all registered and active plugins */
         PluginManager        _pluginManager;

         // Programming model interface
         PMInterface *        _pmInterface;

         //! Enable or disable the use of caches
         bool                 _useCaches;
         //! General cache policy (if not specifically redefined for a certain architecture)
         CachePolicyType      _cachePolicy;
         //! CacheMap register
         CacheMap             _cacheMap;

#ifdef GPU_DEV
         //! Keep record of the data that's directly allocated on pinned memory
         PinnedAllocator      _pinnedMemoryCUDA;
#endif
         std::list<std::string>    _enableEvents;  //FIXME: only in instrumentation
         std::list<std::string>    _disableEvents; //FIXME: only in instrumentation
         std::string               _instrumentDefault; //FIXME: only in instrumentation

         // disable copy constructor & assignment operation
         System( const System &sys );
         const System & operator= ( const System &sys );

         void config ();
         void loadModules();
         void unloadModules();
         
         PE * createPE ( std::string pe_type, int pid );

      public:
         /*! \brief System default constructor
          */
         System ();
         /*! \brief System destructor
          */
         ~System ();

         void start ();
         void finish ();

         int getWorkDescriptorId( void );

         void submit ( WD &work );
         void submitWithDependencies (WD& work, size_t numDataAccesses, DataAccess* dataAccesses);
         void waitOn ( size_t numDataAccesses, DataAccess* dataAccesses);
         void inlineWork ( WD &work );

         void createWD (WD **uwd, size_t num_devices, nanos_device_t *devices,
                        size_t data_size, size_t data_align, void ** data, WG *uwg,
                        nanos_wd_props_t *props, nanos_wd_dyn_props_t *dyn_props, size_t num_copies,
                        nanos_copy_data_t **copies, nanos_translate_args_t translate_args );

         void createSlicedWD ( WD **uwd, size_t num_devices, nanos_device_t *devices, size_t outline_data_size,
                        int outline_data_align, void **outline_data, WG *uwg, Slicer *slicer, nanos_wd_props_t *props, nanos_wd_dyn_props_t *dyn_props,
                        size_t num_copies, nanos_copy_data_t **copies );

         void duplicateWD ( WD **uwd, WD *wd );
         void duplicateSlicedWD ( SlicedWD **uwd, SlicedWD *wd );

        /* \brief prepares a WD to be scheduled/executed.
         * \param work WD to be set up
         */
         void setupWD( WD &work, WD *parent );

         // methods to access configuration variable         
         void setNumPEs ( int npes );

         int getNumPEs () const;
         
         //! \brief Returns the maximum number of threads (SMP + GPU + ...). 
         unsigned getMaxThreads () const; 
         
         void setNumThreads ( int nthreads );
         
         int getNumThreads () const;

         int getCpuId ( int idx ) const;
	 
         int getCpuCount ( ) const;

         void setCpuAffinity(const pid_t pid, size_t cpusetsize, cpu_set_t *mask);

         int checkCpuMask(cpu_set_t *mask);

         void setDeviceStackSize ( int stackSize );

         int getDeviceStackSize () const;

         void setBindingStart ( int value );

         int getBindingStart () const;

         void setBindingStride ( int value );

         int getBindingStride () const;

         void setBinding ( bool set );

         bool getBinding () const;

         ExecutionMode getExecutionMode () const;

         bool getVerbose () const;

         void setVerbose ( bool value );

         void setInitialMode ( InitialMode mode );
         InitialMode getInitialMode() const;

         void setDelayedStart ( bool set);

         bool getDelayedStart () const;

         bool useYield() const;

         int getTaskNum() const;

         int getIdleNum() const;

         int getReadyNum() const;

         int getRunningTasks() const;

         int getNumWorkers() const;

         int getNumWorkers( DeviceData *arch );

         int getNumSockets() const;

         void setNumSockets ( int numSockets );

         int getCoresPerSocket() const;

         void setCoresPerSocket ( int coresPerSocket );
         
         /**
          * \brief Returns a CPU Id that the given architecture should use
          * to tie a new processing element to.
          * \param pe Processing Element number.
          */
         int getBindingId ( int pe ) const;

         void setUntieMaster ( bool value );

         bool getUntieMaster () const;

         void setSynchronizedStart ( bool value );
         bool getSynchronizedStart ( void ) const;

         // team related methods
         BaseThread * getUnassignedWorker ( void );
         ThreadTeam * createTeam ( unsigned nthreads, void *constraints=NULL, bool reuseCurrent=true,
                                   bool enterCurrent=true, bool enterOthers=true, bool starringCurrent = true, bool starringOthers=false );

         BaseThread * getWorker( unsigned int n );

         void endTeam ( ThreadTeam *team );
         void releaseWorker ( BaseThread * thread );

         void setThrottlePolicy( ThrottlePolicy * policy );

         bool throttleTaskIn( void ) const;
         void throttleTaskOut( void ) const;

         const std::string & getDefaultSchedule() const;

         const std::string & getDefaultThrottlePolicy() const;

         const std::string & getDefaultBarrier() const;

         const std::string & getDefaultInstrumentation() const;

         const std::string & getDefaultArch() const;
         void setDefaultArch( const std::string &arch );

         void setHostFactory ( peFactory factory );

         void setDefaultBarrFactory ( barrFactory factory );

         Slicer * getSlicer( const std::string &label ) const;

         WorkSharing * getWorkSharing( const std::string &label ) const;

         Instrumentation * getInstrumentation ( void ) const;

         void setInstrumentation ( Instrumentation *instr );

         void registerSlicer ( const std::string &label, Slicer *slicer);

         void registerWorkSharing ( const std::string &label, WorkSharing *ws);

         void setDefaultSchedulePolicy ( SchedulePolicy *policy );
         
         SchedulePolicy * getDefaultSchedulePolicy ( ) const;

         SchedulerStats & getSchedulerStats ();
         SchedulerConf  & getSchedulerConf();
         
         /*! \brief Disables the execution of pending WDs in the scheduler's
          queue.
         */
         void stopScheduler ();
         /*! \brief Resumes the execution of pending WDs in the scheduler's
          queue.
         */
         void startScheduler ();
         
         //! \brief Checks if the scheduler is stopped or not.
         bool isSchedulerStopped () const;
         
         /*! \brief Waits until all threads are paused. This is useful if you
          * want that no task is executed after the scheduler is disabled.
          * \note The scheduler must be stopped first.
          * \sa stopScheduler(), waitUntilThreadsUnpaused
          */
         void waitUntilThreadsPaused();
         
         /*! \brief Waits until all threads are unpaused. Use this
          * when you require that no task is running in a certain section.
          * In that case, you'll probably disable the scheduler, wait for
          * threads to be paused, do something, and then start over. Before
          * starting over, you need to call this function, because if you don't
          * there is the potential risk of threads been unpaused causing a race
          * condition.
          * \note The scheduler must be started first.
          * \sa stopScheduler(), waitUntilThreadsUnpaused
          */
         void waitUntilThreadsUnpaused();
         
         void pausedThread();
         
         void unpausedThread();
         
         /*! \brief Returns the name of the default dependencies manager.
          */
         const std::string & getDefaultDependenciesManager() const;
         
         /*! \brief Specifies the dependencies manager to be used.
          *  \param manager DependenciesManager.
          */
         void setDependenciesManager ( DependenciesManager *manager );
         
         /*! \brief Returns the dependencies manager in use.
          */
         DependenciesManager * getDependenciesManager ( ) const;

         void setPMInterface (PMInterface *_pm);
         PMInterface & getPMInterface ( void ) const;
         bool isCacheEnabled();
         CachePolicyType getCachePolicy();
         CacheMap& getCacheMap();

#ifdef GPU_DEV
         PinnedAllocator& getPinnedAllocatorCUDA();
#endif

         void threadReady ();

         void registerPlugin ( const char *name, Plugin &plugin );
         bool loadPlugin ( const char *name );
         bool loadPlugin ( const std::string &name );
         Plugin * loadAndGetPlugin ( const char *name );
         Plugin * loadAndGetPlugin ( const std::string &name );
         
         void setValidPlugin ( const std::string &module,  const std::string &plugin );
         
         /*! \brief Registers a plugin option. Depending on whether nanox --help
          * is running or not, it will use a list of valid plugins or not.
          *  \param option Name of the option in NX_ARGS.
          *  \param module Module name (i.e. sched for schedule policies).
          *  \param var Variable that will store the read value (i.e. _defSchedule).
          *  \param helpMessage Help message to be printed in nanox --help
          *  \param cfg Config object.
          */
         void registerPluginOption ( const std::string &option, const std::string &module, std::string &var, const std::string &helpMessage, Config &cfg );
         
                  
         #ifdef MPI_DEV
         void DEEP_Booster_alloc(MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm);  
         #endif
         
   };

   extern System sys;

};

#endif

