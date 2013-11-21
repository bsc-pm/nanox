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
#include "archplugin_decl.hpp"
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
         typedef std::vector<ArchPlugin*> ArchitecturePlugins;
         
         //! CPU id binding list
         typedef std::vector<int> Bindings;
         
         // global seeds
         Atomic<int> _atomicWDSeed; /*!< \brief ID seed for new WD's */
         Atomic<int> _threadIdSeed; /*!< \brief ID seed for new threads */

         // configuration variables
         unsigned int         _numPEs;
         int                  _numThreads;
         int                  _deviceStackSize;
         int                  _bindingStart;
         int                  _bindingStride;
         bool                 _bindThreads;
         bool                 _profile;
         bool                 _instrument;
         bool                 _verboseMode;
         bool                 _summary;            /*!< \brief Flag to enable the summary */
         time_t               _summary_start_time; /*!< \brief Track time to show duration in summary */
         ExecutionMode        _executionMode;
         InitialMode          _initialMode;
         bool                 _untieMaster;
         bool                 _delayedStart;
         bool                 _useYield;
         bool                 _synchronizedStart;
         //! Physical NUMA nodes
         int                  _numSockets;
         int                  _coresPerSocket;
         //! Available NUMA nodes given by the CPU set
         int                  _numAvailSockets;
         //! The socket that will be assigned to the next WD
         int                  _currentSocket;
         //! Enable Dynamic Load Balancing library
         bool                 _enable_dlb;

	 // Nanos++ scheduling domain
         cpu_set_t            _cpu_set;         /*!< \brief system's default cpu_set */
         cpu_set_t            _cpu_active_set;  /*!< \brief mask of current active cpus */

         //! Maps from a physical NUMA node to a user-selectable node
         std::vector<int>     _numaNodeMap;

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
         
         /*! Architecture plugins */
         ArchitecturePlugins  _archs;
         

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
         
         //! CPU id binding list
         Bindings             _bindings;
         
         //! hwloc topology structure
         void *               _hwlocTopology;
         //! Path to a hwloc topology xml
         std::string          _topologyPath;

#ifdef GPU_DEV
         //! Keep record of the data that's directly allocated on pinned memory
         PinnedAllocator      _pinnedMemoryCUDA;
#endif
#ifdef NANOS_INSTRUMENTATION_ENABLED
         std::list<std::string>    _enableEvents;
         std::list<std::string>    _disableEvents;
         std::string               _instrumentDefault;
         bool                      _enable_cpuid_event;
#endif

         const int                 _lockPoolSize;
         Lock *                    _lockPool;
         ThreadTeam               *_mainTeam;

         // disable copy constructor & assignment operation
         System( const System &sys );
         const System & operator= ( const System &sys );

         void config ();
         void loadModules();
         void unloadModules();

         /*!
          * \brief Creates a new PE and a new thread associated to it
          * \param[in] p ID of the new PE
          */
         void createWorker( unsigned p );

         /*!
          * \brief Set up the teamData of the thread to be included in the team, and optionally add it
          * \param[in,out] team The team where the thread will be added
          * \param[in,out] thread The thread to be included
          * \param[in] enter Should the thread enter the team?
          * \param[in] star Is the thread a star within the team?
          * \param[in] creator Is the thread the creator of the team?
          */
         void acquireWorker( ThreadTeam * team, BaseThread * thread, bool enter=true, bool star=false, bool creator=false );

         /*!
          * \brief Updates team members so that it matches with system's _cpu_active_set
          */
         void applyCpuMask();

         /*!
          * \brief Processes the system's _cpu_active_set for later update the threads
          *
          * Depending on the system binding configuration, this function will update _bindings to be able
          * later to create new PE's or just update the raw number of threads if binding is disabled
          */
         void processCpuMask( void );
         
         void loadHwloc();
         void unloadHwloc();
         
         PE * createPE ( std::string pe_type, int pid, int uid );

         //* \brief Prints the Environment Summary (resources, plugins, prog. model, etc.) before the execution
         void environmentSummary( void );

         //* \brief Prints the Execution Summary (time, completed tasks, etc.) at the end of the execution
         void executionSummary( void );

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
                        nanos_wd_props_t *props, nanos_wd_dyn_props_t *dyn_props, size_t num_copies, nanos_copy_data_t **copies,
                        size_t num_dimensions, nanos_region_dimension_internal_t **dimensions,
                        nanos_translate_args_t translate_args, const char *description );

         void createSlicedWD ( WD **uwd, size_t num_devices, nanos_device_t *devices, size_t outline_data_size,
                        int outline_data_align, void **outline_data, WG *uwg, Slicer *slicer, nanos_wd_props_t *props, nanos_wd_dyn_props_t *dyn_props,
                        size_t num_copies, nanos_copy_data_t **copies, size_t num_dimensions, nanos_region_dimension_internal_t **dimensions, const char *description );

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

         int getCpuCount ( ) const;

         /*!
          * \brief Get current system's _cpu_active_set
          * \param[out] mask
          */
         void getCpuMask ( cpu_set_t *mask ) const;

         /*!
          * \brief Set current system's _cpu_active_set
          * \param[in] mask
          */
         void setCpuMask ( const cpu_set_t *mask );

         /*!
          * \brief Add mas to the current system's _cpu_active_set
          * \param[in] mask
          */
         void addCpuMask ( const cpu_set_t *mask );

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

         int getCreatedTasks() const ;

         int getTaskNum() const;

         int getIdleNum() const;

         int getReadyNum() const;

         int getRunningTasks() const;

         int getNumWorkers() const;

         int getNumWorkers( DeviceData *arch );

         /** \brief Returns the number of physical NUMA nodes. */
         int getNumSockets() const;

         void setNumSockets ( int numSockets );

         /** \brief Returns the number of NUMA nodes available for the user. */
         int getNumAvailSockets() const;

         /**
          * \brief Translates from a physical NUMA node to a virtual (user-selectable) node.
          * \return A number in the range [0..N) where N is the number of virtual NUMA nodes,
          * or INT_MIN if that physical node cannot be used.
          */
         int getVirtualNUMANode( int physicalNode ) const;

         int getCurrentSocket() const;

         /**
          * \brief Sets the (virtual) node where tasks should be executed.
          * \param currentSocket A value in the range [0,N) where N is the number
          * of available nodes (what is returned by getNumAvailSockets()).
          * \see getNumAvailSockets.
          */
         void setCurrentSocket( int currentSocket );

         int getCoresPerSocket() const;

         void setCoresPerSocket ( int coresPerSocket );
         
         /**
          * \brief Returns a CPU Id that the given architecture should use
          * to tie a new processing element to.
          * \param pe Processing Element number.
          * \note This method is the one that uses the affinity mask and binding
          * start and stride parameters.
          */
         int getBindingId ( int pe ) const;
         
         /**
          * \brief Reserves a PE to be used exclusively by a certain
          * architecture.
          * If you try to reserve all PEs, leaving no PEs for SMPs, reserved
          * will be false and a warning will be displayed.
          * \param reserveNode [in] If enabled, will try to reserve the PE in
          * the node specified by the node parameter, otherwise, that parameter
          * will be ignored.
          * \param node [in] NUMA node to reserve the PE from. It is only used
          * when reserveNode is true.
          * \param reserved [out] If the PE was successfully reserved or not.
          * \return Id of the PE to reserve.
          */
         unsigned reservePE ( bool reserveNode, unsigned node, bool & reserved );
         
         /**
          * \brief Checks if hwloc is available.
          */
         bool isHwlocAvailable () const;
         
         /**
          * \brief Returns the hwloc_topology_t structure.
          * This structure will only be available for a short window during
          * System::start. Otherwise, NULL will be returned.
          * In order to avoid surrounding this function by ifdefs, it returns
          * a void * that you must cast to hwloc_topology_t.
          */
         void * getHwlocTopology ();
         
         /*!
          * \brief Sets the number of NUMA nodes and the number of cores per
          * NUMA node .
          * Uses hwloc if available.
          */
         void loadNUMAInfo ();
         
         /*!
          * \brief Sets the the number of active/available NUMA nodes.
          * Creates the NUMA node translation table as well.
          * \note It is really important to call this after PEs are created.
          */
         void completeNUMAInfo ();

         /** \brief Retrieves the NUMA node of a given PE.
          *  \note Will use hwloc if available.
          */
         unsigned getNodeOfPE ( unsigned pe );

         void setUntieMaster ( bool value );

         bool getUntieMaster () const;

         void setSynchronizedStart ( bool value );
         bool getSynchronizedStart ( void ) const;

         int nextThreadId ();

         /*!
          * \brief Returns whether DLB is enabled or not
          */
         bool dlbEnabled() const;

         // team related methods
         /*!
          * \brief Returns, if any, the worker thread with lower ID that has no team or that has been tagged to sleep
          */
         BaseThread * getUnassignedWorker ( void );

         /*!
          * \brief Returns, if any, the worker thread with upper ID that has team and still has not been tagged to sleep
          */
         BaseThread * getAssignedWorker ( void );

         /*!
          * \brief Returns a new created Team with the specified parameters
          * \param[in] nthreads The team size
          * \param[in] constraints Not used
          * \param[in] reuseCurrent Will this thread be a member of the team?
          * \param[in] enterCurrent Will this thread immediately enter the team?
          * \param[in] enterOthers Will the other threads immediately enter the team?
          * \param[in] starringCurrent Is this a star thread?
          * \param[in] starringOthers Are the others star threads?
          */
         ThreadTeam * createTeam ( unsigned nthreads, void *constraints=NULL, bool reuseCurrent=true,
                                   bool enterCurrent=true, bool enterOthers=true, bool starringCurrent = true, bool starringOthers=false );

         BaseThread * getWorker( unsigned int n );

         void endTeam ( ThreadTeam *team );

         /*!
          * \brief Releases a worker thread from its team
          * \param[in,out] thread
          */
         void releaseWorker ( BaseThread * thread );

         /*!
          * \brief Updates the number of active worker threads and adds them to the main team
          * \param[in] nthreads
          */
         void updateActiveWorkers ( int nthreads );

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

#ifdef NANOS_INSTRUMENTATION_ENABLED
         bool isCpuidEventEnabled ( void ) const;
#endif

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
         
         /**! \brief Register an architecture plugin.
          *   \param plugin A pointer to the plugin.
          *   \return The index of the plugin in the vector.
          */
         size_t registerArchitecture( ArchPlugin * plugin );

#ifdef GPU_DEV
         char * getOmpssUsesCuda();
         char * getOmpssUsesCublas();

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

         /*! \brief Returns one of the system lock (belonging to the pool of locks)
          */
         Lock * getLockAddress(void *addr ) const;

         /*! \brief Returns if there are pendant writes for a given memory address
          *
          *  \param [in] addr memory address
          *  \return {True/False} depending if there are pendant writes
          */
         bool haveDependencePendantWrites ( void *addr ) const;

         /*! \brief Active current thread (i.e. pthread ) and include it into the main team
          */
         void admitCurrentThread ( void );
         void expelCurrentThread ( void );
   };

   extern System sys;

};

#endif

