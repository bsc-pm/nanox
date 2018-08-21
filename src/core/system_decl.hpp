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
#include "network_decl.hpp"
#include "pminterface_decl.hpp"
#include "plugin_decl.hpp"
#include "archplugin_decl.hpp"
#include "barrier_decl.hpp"
#include "accelerator_decl.hpp"
#include "location_decl.hpp"
#include "addressspace_decl.hpp"
#include "smpbaseplugin_decl.hpp"
#include "hwloc_decl.hpp"
#include "threadmanager_decl.hpp"
#include "router_decl.hpp"

#include "regiondirectory_decl.hpp"
#include "smpdevice_decl.hpp"

#ifdef GPU_DEV
#include "pinnedallocator_decl.hpp"
#include "gpuprocessor_fwd.hpp"
#endif

#ifdef OpenCL_DEV
#include "openclprocessor_fwd.hpp"
#endif

#ifdef CLUSTER_DEV
#include "clustermpiplugin_fwd.hpp"
#else
namespace nanos {
namespace ext {
class ClusterMPIPlugin;
}
}
#endif

namespace nanos {

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

         typedef void (*Init) ();
         //typedef std::vector<Accelerator *> AList;

         typedef std::map<unsigned int, BaseThread *> ThreadList;

      private:
         // types
         typedef std::map<std::string, Slicer *> Slicers;
         typedef std::map<std::string, WorkSharing *> WorkSharings;
         typedef std::multimap<std::string, std::string> ModulesPlugins;
         typedef std::vector<ArchPlugin*> ArchitecturePlugins;

         //! \brief Compiler supplied flags in symbols
         struct SuppliedFlags
         {
            //! If the program is using priorities
            bool prioritiesNeeded : 1;
         };

         SuppliedFlags        _compilerSuppliedFlags; //!< \brief Compiler supplied flags

         // global seeds
         Atomic<int> _atomicWDSeed;                   //!< \brief ID seed for new WD's
         Atomic<int> _threadIdSeed;                   //!< \brief ID seed for new threads
         Atomic<unsigned int> _peIdSeed;              //!< \brief ID seed for new PE's

         // Devices
         SMPDevice _SMP;

         // configuration variables
         size_t               _deviceStackSize;
         bool                 _profile;
         bool                 _instrument;
         bool                 _verboseMode;
         bool                 _summary;               //!< \brief Flag to enable the summary
         time_t               _summaryStartTime;      //!< \brief Track time to show duration in summary
         ExecutionMode        _executionMode;
         InitialMode          _initialMode;
         bool                 _untieMaster;
         bool                 _delayedStart;
         bool                 _synchronizedStart;
         bool                 _alreadyFinished;       //!< \brief Prevent System::finish from being executed more than once.
         bool                 _predecessorLists;      //!< \brief Maintain predecessors list (disabled by default).


         ThrottlePolicy      *_throttlePolicy;
         SchedulerStats       _schedStats;
         SchedulerConf        _schedConf;
         std::string          _defSchedule;           //!< \brief Name of default scheduler
         std::string          _defThrottlePolicy;     //!< \brief Name of default throttole policy (cutoff)
         std::string          _defBarr;               //!< \brief Name of default barrier
         std::string          _defInstr;              //!< \brief Name of default instrumentation
         std::string          _defDepsManager;        //!< \brief Name of default dependences manager
         std::string          _defArch;               //!< \brief Name of default architercture
#if 0 /* _defDeviceName and _defDevice seem unused */
         std::string          _defDeviceName;         //!< \brief Name of default device
         const Device         *_defDevice;
#endif

         /*! factories for scheduling, pes and barriers objects */
         peFactory            _hostFactory;
         barrFactory          _defBarrFactory;

         /*! Valid plugin map (module)->(list of plugins) */
         ModulesPlugins       _validPlugins;

         /*! Architecture plugins */
         SMPBasePlugin       *_smpPlugin;
         ArchitecturePlugins  _archs;


         PEMap                _pes;
         ThreadList           _workers;

         //! List of all supported architectures by _pes
         DeviceList           _devices;

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

         /*! Cluster: system Network object */
         Network              _net;
         bool                 _usingCluster;
         bool                 _usingClusterMPI;
         ext::ClusterMPIPlugin *_clusterMPIPlugin;
         bool                 _usingNode2Node;
         bool                 _usingPacking;
         std::string          _conduit;

         WorkSharings         _worksharings; /**< set of global worksharings */

         Instrumentation     *_instrumentation; /**< Instrumentation object used in current execution */
         SchedulePolicy      *_defSchedulePolicy;

         /*! Dependencies domain manager */
         DependenciesManager *_dependenciesManager;

         /*! It manages all registered and active plugins */
         PluginManager        _pluginManager;

         // Programming model interface
         PMInterface *        _pmInterface;

         WD *slaveParentWD;
         BaseThread *_masterGpuThd;

         unsigned int                                  _separateMemorySpacesCount;
         std::vector< SeparateMemoryAddressSpace * >   _separateAddressSpaces;
         HostMemoryAddressSpace                        _hostMemory;
         RegionCache::CachePolicy                      _regionCachePolicy;
         std::string                                   _regionCachePolicyStr;
         std::size_t                                   _regionCacheSlabSize;

         std::set<unsigned int>                        _clusterNodes;
         std::set<unsigned int>                        _numaNodes;
         std::set<memory_space_id_t>                   _activeMemorySpaces;

         unsigned int                                  _acceleratorCount;
         //! Maps from a physical NUMA node to a user-selectable node
         std::vector<int>                              _numaNodeMap;

         /*! Thread Manager members */
         ThreadManagerConf                             _threadManagerConf;
         ThreadManager *                               _threadManager;

#ifdef GPU_DEV
         //! Keep record of the data that's directly allocated on pinned memory
         PinnedAllocator      _pinnedMemoryCUDA;
#endif
#ifdef NANOS_INSTRUMENTATION_ENABLED
         std::list<std::string>    _enableEvents;
         std::list<std::string>    _disableEvents;
         std::string               _instrumentDefault;
         bool                      _enableCpuidEvent;
#endif

         const int                 _lockPoolSize;
         Lock *                    _lockPool;
         ThreadTeam               *_mainTeam;
         bool                      _simulator;

         //! Specifies the maximum number of times a recoverable task can re-execute (avoids infinite recursion).
         int                       _task_max_retries;

         // disable copy constructor & assignment operation
         System( const System &sys );
         const System & operator= ( const System &sys );

         //! \brief Reads environment variables and compiler-supplied flags
         void config ();
         void loadModules();
         void loadArchitectures();
         void unloadModules();

         Atomic<int> _atomicSeedWg;
         Atomic<unsigned int> _affinityFailureCount;
         bool                      _createLocalTasks;
         bool _verboseDevOps;
         bool _verboseCopies;
         bool _splitOutputForThreads;
         int _userDefinedNUMANode;
         Router _router;
         Lock _allocLock;
      public:
         Hwloc _hwloc;
         bool _immediateSuccessorDisabled;
         bool _predecessorCopyInfoDisabled;
         bool _invalControl;
         bool _cgAlloc;
         bool _inIdle;
         bool _lazyPrivatizationEnabled;
         bool _preSchedule;
         std::map<int, std::set<WD *> > _slots;
         void *_watchAddr;

      private:
         PE * createPE ( std::string pe_type, int pid, int uid );

         /*! \brief Prints the Environment Summary (resources, plugins, prog. model, etc.)
          */
         void environmentSummary( void );

         /*! \brief Prints the Execution Summary (time, completed tasks, etc.)
          */
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


         /*!
          * \brief Set up the teamData of the thread to be included in the team, and optionally add it
          * \param[in,out] team The team where the thread will be added
          * \param[in,out] thread The thread to be included
          * \param[in] enter Should the thread enter the team?
          * \param[in] star Is the thread a star within the team?
          * \param[in] creator Is the thread the creator of the team?
          */
         void acquireWorker( ThreadTeam * team, BaseThread * thread, bool enter=true, bool star=false, bool creator=false );


         void submit ( WD &work );
         void submitWithDependencies (WD& work, size_t numDataAccesses, DataAccess* dataAccesses);
         void waitOn ( size_t numDataAccesses, DataAccess* dataAccesses);
         void inlineWork ( WD &work );

         void createWD (WD **uwd, size_t num_devices, nanos_device_t *devices,
                        size_t data_size, size_t data_align, void ** data, WD *uwg,
                        nanos_wd_props_t *props, nanos_wd_dyn_props_t *dyn_props, size_t num_copies, nanos_copy_data_t **copies,
                        size_t num_dimensions, nanos_region_dimension_internal_t **dimensions,
                        nanos_translate_args_t translate_args, const char *description, Slicer *slicer );

         void duplicateWD ( WD **uwd, WD *wd );

        /* \brief prepares a WD to be scheduled/executed.
         * \param work WD to be set up
         */
         void setupWD( WD &work, WD *parent );

         /*!
          * \brief Method to get the device types of all the architectures running
          */
         DeviceList & getSupportedDevices();

         void setDeviceStackSize ( size_t stackSize );

         size_t getDeviceStackSize () const;

         ExecutionMode getExecutionMode () const;

         bool getVerbose () const;

         void setVerbose ( bool value );

         void setInitialMode ( InitialMode mode );
         InitialMode getInitialMode() const;

         void setDelayedStart ( bool set);

         bool getDelayedStart () const;

         int getCreatedTasks() const ;

         int getTaskNum() const;

         int getIdleNum() const;

         int getReadyNum() const;

         int getRunningTasks() const;

         int getNumWorkers() const;

         int getNumCreatedPEs() const;

         int getNumWorkers( DeviceData *arch );

         int getNumThreads() const;

         void setUntieMaster ( bool value );

         bool getUntieMaster () const;

         void setSynchronizedStart ( bool value );
         bool getSynchronizedStart ( void ) const;

         //! \brief Enables or disables the use of predecessor lists
         void setPredecessorLists ( bool value );
         //! \brief Checks if predecessor lists are enabled
         bool getPredecessorLists ( void ) const;

         int nextThreadId ();
         unsigned int nextPEId ();

         /*!
          * \brief Returns the maximum number of times a task can try to recover from an error by re-executing itself.
          */
         int getTaskMaxRetries() const;

         // team related methods
         /*!
          * \brief Returns, if any, the worker thread with lower ID that has no team or that has been tagged to sleep
          */
         BaseThread * getUnassignedWorker ( void );

         /*!
          * \brief Returns a new team of threads
          * \param[in] nthreads Number of threads in the team.
          * \param[in] constraints This parameter is not used.
          * \param[in] reuse Reuse current thread as part of the team.
          * \param[in] parallel Identifies the type of team, parallel code or single executor.
          */
         ThreadTeam * createTeam ( unsigned nthreads, void *constraints=NULL, bool reuse=true, bool enter=true, bool parallel=false );

         ThreadList::iterator getWorkersBegin();
         ThreadList::iterator getWorkersEnd();

         BaseThread * getWorker( unsigned int n );

         void endTeam ( ThreadTeam *team );

         /*!
          * \brief Updates the number of active worker threads and adds them to the main team
          * \param[in] nthreads
          */
         void updateActiveWorkers( int nthreads );

         /*!
          * \brief Get the process mask of active CPUs by reference
          */
         const CpuSet& getCpuProcessMask() const;

         /*!
          * \brief Set the process mask
          * \param[in] mask
          * \return True if the mask was completely set,
          *          False if the mask was either invalid or only partially set
          */
         bool setCpuProcessMask( const CpuSet& mask );

         /*!
          * \brief Add the CPUs in mask into the current process mask
          * \param[in] mask
          */
         void addCpuProcessMask( const CpuSet& mask );

         /*!
          * \brief Get the current mask of active CPUs by reference
          */
         const CpuSet& getCpuActiveMask() const;

         /*!
          * \brief Set the mask of active CPUs
          * \param[in] mask
          * \return True if the mask was completely set,
          *          False if the mask was either invalid or only partially set
          */
         bool setCpuActiveMask( const CpuSet& mask );

         /*!
          * \brief Add the CPUs in mask into the current mask of active CPUs
          * \param[in] mask
          */
         void addCpuActiveMask( const CpuSet& mask );

         /*!
          * \brief Enable one CPU in the active cpu mask
          * \param[in] cpuid CPU id to enable
          */
         void enableCpu( int cpuid );

         /*!
          * \brief Disable one CPU in the active cpu mask
          * \param[in] cpuid CPU id to disable
          */
         void disableCpu( int cpuid );

         /*!
          * \brief Force the creation of at least 1 thread per PE, which are blocked afterwards
          */
         void forceMaxThreadCreation();

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

         Network * getNetwork( void );
         bool usingCluster( void ) const;
         bool usingClusterMPI( void ) const;
         bool usingNewCache( void ) const;
         bool useNode2Node( void ) const;
         bool usePacking( void ) const;
         const std::string & getNetworkConduit() const;

         void stopFirstThread( void );

         void setPMInterface (PMInterface *_pm);
         PMInterface & getPMInterface ( void ) const;
         bool isCacheEnabled();

         /**! \brief Register an architecture plugin.
          *   \param plugin A pointer to the plugin.
          *   \return The index of the plugin in the vector.
          */
         size_t registerArchitecture( ArchPlugin * plugin );

#ifdef GPU_DEV
         PinnedAllocator& getPinnedAllocatorCUDA();
#endif

         void threadReady ();

         void setSlaveParentWD( WD * wd ){ slaveParentWD = wd ; };
         WD* getSlaveParentWD( ){ return slaveParentWD ; };

         void registerPlugin ( const char *name, Plugin &plugin );
         bool loadPlugin ( const char *name );
         bool loadPlugin ( const std::string &name );
         Plugin * loadAndGetPlugin ( const char *name );
         Plugin * loadAndGetPlugin ( const std::string &name );
         int getWgId();
         unsigned int getRootMemorySpaceId();

         HostMemoryAddressSpace &getHostMemory() { return _hostMemory; }

         SeparateMemoryAddressSpace &getSeparateMemory( memory_space_id_t id ) {
            //std::cerr << "Requested object " << _separateAddressSpaces[ id ] <<std::endl;
            return *(_separateAddressSpaces[ id ]);
         }

         void addSeparateMemory( memory_space_id_t id, SeparateMemoryAddressSpace* memory) {
            //std::cerr << "Requested object " << _separateAddressSpaces[ id ] <<std::endl;
            _separateAddressSpaces[ id ]=memory;
         }

         unsigned int getNewSeparateMemoryAddressSpaceId() { return _separateMemorySpacesCount++; }
         unsigned int getSeparateMemoryAddressSpacesCount() { return _separateMemorySpacesCount - 1; }

      //private:
         //std::list< std::list<GraphEntry *> * > _graphRepLists;
         //Lock _graphRepListsLock;
      public:
         //std::list<GraphEntry *> *getGraphRepList();

         RegionDirectory const &getMasterRegionDirectory() { return _hostMemory.getDirectory(); }
         ProcessingElement &getPEWithMemorySpaceId( memory_space_id_t id );

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


        /**
         * \brief Registers PEs to current nanox workers/team
         * Function created to serve MPI device
         * Whoever creates the threads is reponsible
         * of increasing extrae max threads
         * \param num_pes number of process spawned
         * \param pes pointer to a list of Processing Elements
         */
         void addPEsAndThreadsToTeam(PE **pes, int num_pes, BaseThread** threads, int num_threads);

         void increaseAffinityFailureCount() { _affinityFailureCount++; }
         unsigned int getAffinityFailureCount() { return _affinityFailureCount.value(); }

         /*! \brief Active current thread (i.e. pthread ) and include it into the main team
          */
         void admitCurrentThread ( bool isWorker );
         void expelCurrentThread ( bool isWorker );

         /*! \brief Setup of the runtime at the beginning of the top level function of the program
          *
          * Some devices (like MPI and cluster) may require extra
          * initialization steps, this function performs them.
          *
          * Also resilency support uses this function to set up signal handlers.
          *
          * Under instrumentation, this function emits an initial event that it
          * is used to track the entry point of the program.
          */
         void ompss_nanox_main(void *addr, const char* file, int line);

         /*! \brief Shutdown notification of leaving the top level function of the program
          *
          * This function is typically called from an atexit handler and
          * currently it only emits an event to track finalization of the
          * program.
          */
         void ompss_nanox_main_end ();

         global_reg_t _registerMemoryChunk(void *addr, std::size_t len);
         void registerNodeOwnedMemory(unsigned int node, void *addr, std::size_t len);
         void stickToProducer(void *addr, std::size_t len);
         void setCreateLocalTasks(bool value);
         memory_space_id_t addSeparateMemoryAddressSpace( Device &arch, bool allocWide, std::size_t slabSize );
         void setSMPPlugin(SMPBasePlugin *p);
         SMPBasePlugin *getSMPPlugin() const;
         bool isSimulator() const;
         ThreadTeam *getMainTeam();
         bool getVerboseDevOps() const;
         void setVerboseDevOps(bool value);
         bool getVerboseCopies() const;
         void setVerboseCopies(bool value);
         bool getSplitOutputForThreads() const;
         std::string getRegionCachePolicyStr() const;
         void setRegionCachePolicyStr( std::string policy );
         RegionCache::CachePolicy getRegionCachePolicy() const;
         std::size_t getRegionCacheSlabSize() const;
         void createDependence( WD* pred, WD* succ);
         unsigned int getNumClusterNodes() const;
	 unsigned int getNumNumaNodes() const;
	 //! Return a vector which maps physical NUMA nodes (vector indexes)
	 //! with virtual nodes (vector values)
	 const std::vector<int>& getNumaNodeMap() const;
         //! Return INT_MIN if physicalNode does not have a mapping.
         int getVirtualNUMANode( int physicalNode ) const;
         std::set<unsigned int> const &getClusterNodeSet() const;
         memory_space_id_t getMemorySpaceIdOfClusterNode( unsigned int node ) const;
         int getUserDefinedNUMANode() const;
         void setUserDefinedNUMANode( int nodeId );
         void registerObject( int numObjects, nanos_copy_data_internal_t *obj );
         void unregisterObject( int numObjects, void *base_addresses );

         unsigned int getNumAccelerators() const;
         unsigned int getNewAcceleratorId();
         memory_space_id_t getMemorySpaceIdOfAccelerator( unsigned int acceleratorId ) const;

         ThreadManagerConf& getThreadManagerConf();
         ThreadManager* getThreadManager() const;

         //! \brief Returns true if the compiler says priorities are required
         bool getPrioritiesNeeded() const;
         Router& getRouter();
         void switchToThread( unsigned int thid );
         bool isImmediateSuccessorEnabled() const;
         bool usePredecessorCopyInfo() const;
         bool invalControlEnabled() const;
         std::set<memory_space_id_t> const &getActiveMemorySpaces() const;
         PEMap& getPEs();
         void allocLock();
         void allocUnlock();
         bool useFineAllocLock() const;
void _distributeObject( global_reg_t &reg, unsigned int start_node, std::size_t num_nodes );
global_reg_t _registerMemoryChunk_2dim(void *addr, std::size_t rows, std::size_t cols, std::size_t elem_size);

         SMPDevice &_getSMPDevice();
         int initClusterMPI(int *argc, char ***argv);
         void finalizeClusterMPI();
         void notifyIntoBlockingMPICall();
         void notifyOutOfBlockingMPICall();
         void notifyIdle( unsigned int node );
         void disableHelperNodes();
         void preSchedule();
   };

   extern System sys;

} // namespace nanos

#endif
