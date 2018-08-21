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

#ifndef _NANOS_WORK_DESCRIPTOR_DECL_H
#define _NANOS_WORK_DESCRIPTOR_DECL_H

#include <stdlib.h>
#include <utility>
#include <vector>

#include "workdescriptor_fwd.hpp"
#include "slicer_fwd.hpp"
#include "basethread_fwd.hpp"
#include "processingelement_fwd.hpp"
#include "wddeque_fwd.hpp"

#include "dependableobjectwd_decl.hpp"
#include "copydata_decl.hpp"
#include "synchronizedcondition_decl.hpp"
#include "atomic_decl.hpp"
#include "lazy_decl.hpp"
#include "instrumentationcontext_decl.hpp"
#include "compatibility.hpp"

#include "regioncache_decl.hpp"
#include "memcontroller_decl.hpp"

#include "dependenciesdomain_decl.hpp"
#include "task_reduction_decl.hpp"
#include "simpleallocator_decl.hpp"
#include "schedule_fwd.hpp"   // ScheduleWDData

namespace nanos {

typedef std::set<const Device *>  DeviceList;

   /*! \brief This class represents a device object
    */
   class Device
   {
      private:

         const char *_name; /**< Identifies device type */
         Atomic<unsigned int> _numOps;

      public:

         /*! \brief Device constructor
          */
         Device ( const char *n ) : _name ( n ), _numOps(0) {}

         /*! \brief Device copy constructor
          */
         Device ( const Device &arch ) : _name ( arch._name ), _numOps( arch._numOps ) {}

         /*! \brief Device destructor
          */
         ~Device() {};

         /*! \brief Device assignment operator
          */
         const Device & operator= ( const Device &arch ) { _name = arch._name; _numOps = arch._numOps; return *this; }

         /*! \brief Device equals operator
          */
         //bool operator== ( const Device &arch ) { return ( 0 == std::strcmp( arch._name , _name ) ); }
         bool operator== ( const Device &arch ) const { return arch._name == _name; }

         /*! \brief Get device name
          */
         const char * getName ( void ) const { return _name; }
         unsigned int increaseNumOps() { return _numOps++; }

         virtual void *memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, WorkDescriptor const *wd, unsigned int copyIdx) = 0;
         virtual void memFree( uint64_t addr, SeparateMemoryAddressSpace &mem ) = 0;
         virtual void _canAllocate( SeparateMemoryAddressSpace &mem, std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes ) = 0;
         virtual std::size_t getMemCapacity( SeparateMemoryAddressSpace &mem ) = 0;

         virtual void _copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WorkDescriptor const *wd, void *hostObject, reg_t hostRegionId ) = 0;
         virtual void _copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WorkDescriptor const *wd, void *hostObject, reg_t hostRegionId ) = 0;
         virtual bool _copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memorig, DeviceOps *ops, WorkDescriptor const *wd, void *hostObject, reg_t hostRegionId ) = 0;
         virtual void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WorkDescriptor const *wd, void *hostObject, reg_t hostRegionId ) = 0;
         virtual void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WorkDescriptor const *wd, void *hostObject, reg_t hostRegionId ) = 0;
         virtual bool _copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WorkDescriptor const *wd, void *hostObject, reg_t hostRegionId ) = 0;
         virtual void _getFreeMemoryChunksList( SeparateMemoryAddressSpace &mem, SimpleAllocator::ChunkList &list ) = 0;
   };

  /*! \brief This class holds the specific data for a given device
   *
   */
   class DeviceData
   {
      public:
         typedef void ( *work_fct ) ( void *self );
      protected:
         /**Use pointers for this as is this fastest way to compare architecture compatibility */
         const Device *_architecture; /**< Related Device (architecture). */
      private:
         work_fct       _work;

      public:

         /*! \brief DeviceData constructor
          */
         DeviceData ( const Device *arch, work_fct w ) : _architecture ( arch ), _work(w) {}

         /*! \brief DeviceData copy constructor
          */
         DeviceData ( const DeviceData &dd ) : _architecture ( dd._architecture ), _work( dd._work )  {}

         /*! \brief DeviceData destructor
          */
         virtual ~DeviceData() {}

         /*! \brief DeviceData assignment operator
          */
         const DeviceData & operator= ( const DeviceData &dd )
         {
            // self-assignment: ok
            _architecture = dd._architecture;
            _work         = dd._work;
            return *this;
         }

         /*! \brief Returns the device associated to this DeviceData
          *
          *  \return the Device pointer.
          */
         const Device * getDevice () const;

         //! \brief Retuns work
         work_fct getWorkFct() const;

         /*! \brief Indicates if DeviceData is compatible with a given Device
          *
          *  \param[in] arch is the Device which we have to compare to.
          *  \return a boolean indicating if both elements (DeviceData and Device) are compatible.
          */
         bool isCompatible ( const Device &arch ) ;

         /*! \brief FIXME: (#170) documentation needed
          */
         virtual void lazyInit (WorkDescriptor &wd, bool isUserLevelThread, WorkDescriptor *previous=NULL ) = 0;

         /*! \brief FIXME: (#170) documentation needed
          */
         virtual size_t size ( void ) = 0;

         /*! \brief FIXME: (#170) documentation needed
          */
         virtual DeviceData *copyTo ( void *addr ) = 0;
         const char * getName ( void ) const { return _architecture->getName(); }

         virtual DeviceData *clone () const = 0;

    };

/*! \brief This class identifies a single unit of work
 *
 * A slicible WorkDescriptor defines an specific behaviour which potentially can divide a WorkDescriptor
 * in smaller WorkDescriptor's
 *
 * The main idea behind this behaviour is to offer a mechanism which allow to decompose a WorkDescriptor in a
 * set of several WorkDescriptors. Initial implementation of this mechanism is related with the
 * ticket:96.
 *
 * A slicible WordDescriptor will be always related with:
 *
 * - a Slicer, which defines the work descriptor behaviour.
 * - a SlicerData, which keeps all the data needed for splitting the work.
 * - Slicer objects are common for all the slicible WordDescriptor of an specific type. In fact, the Slicer object
 *   determines the type of the slicible WordDescriptor. In the other hand, SlicerData objects are individual for
 *   each slicible WordDescriptor object.
 *
 * A slicible WordDescriptor modifies the behaviour of submit() and dequeue() methods.
 *
 * The common behaviour of a WorkDescriptor submit() method just call Scheduller::submit() method and dequeue() returns
 * the WD itself (meaning this is the work unit ready to be executed) and a boolean value (true,
 * meaning that it will be the last execution for this unit of work). Otherwise, slicible WordDescriptor will execute
 * Slicer::submit() and Slicer::dequeue() respectively, giving the slicer the responsibility of doing specific actions
 * at submission or dequeuing time.
 *
 */
   class WorkDescriptor
   {
      public: /* types */
         typedef enum { IsNotAUserLevelThread=false, IsAUserLevelThread=true } ULTFlag;
         typedef std::vector<WorkDescriptor **> WorkDescriptorPtrList;
         typedef TR1::unordered_map<void *, TR1::shared_ptr<WorkDescriptor *> > CommutativeOwnerMap;
         typedef struct {
            bool is_final;         //!< Work descriptor will not create more work descriptors
            bool is_initialized;   //!< Work descriptor is initialized
            bool is_started;       //!< Work descriptor has been already started
            bool is_ready;         //!< Work descriptor is ready to execute
            bool to_tie;           //!< Work descriptor should to be tied to first thread executing it
            bool is_submitted;     //!< Has this WD been submitted to the Scheduler?
            bool is_configured;    //!< Has this WD been configured to the Scheduler?
            bool is_implicit;        //!< Is the WD an implicit task (in a team)?
            bool is_recoverable;   //!< Flags a task as recoverable, that is, it can be re-executed if it finished with errors.
            bool is_invalid;       //!< Flags an invalid workdescriptor. Used in resiliency when a task fails.
            bool is_runtime_task;  //!< Is the WD a task for doing runtime jobs?
         } WDFlags;
         typedef enum { INIT, START, READY, BLOCKED } State;
         typedef int PriorityType;
         typedef SingleSyncCond<EqualConditionChecker<int> >  components_sync_cond_t;
         typedef std::vector<TaskReduction *>        task_reduction_vector_t;  //< List of task reductions type
      private: /* data members */
         int                           _id;                     //!< Work descriptor identifier
         int                           _hostId;                 //!< Work descriptor identifier @ host
         Atomic<int>                   _components;             //!< Number of components (children, direct descendants)
         components_sync_cond_t        _componentsSyncCond;     //!< Synchronize condition on components
         WorkDescriptor               *_parent;                 //!< Parent WD in task hierarchy
         WorkDescriptor               *_forcedParent;           //!< Forced parent, it will be not notified when finishing
         size_t                        _data_size;              //!< WD data size
         size_t                        _data_align;             //!< WD data alignment
         void                         *_data;                   //!< WD data
         size_t                        _totalSize;              //!< Chunk total size, when allocating WD + extra data
         void                         *_wdData;                 //!< Internal WD data. Allowing higher layer to associate data to WD
         ScheduleWDData               *_scheduleData;           //!< Data set by the scheduling policy
         WDFlags                       _flags;                  //!< WD Flags
         BaseThread                   *_tiedTo;                 //!< Thread is tied to base thread
         memory_space_id_t             _tiedToLocation;         //!< Thread is tied to a memory location
         State                         _state;                  //!< Workdescriptor current state
         GenericSyncCond              *_syncCond;               //!< Generic synchronize condition
         WDPool                       *_myQueue;                //!< Allows dequeuing from third party (e.g. Cilk schedulers)
         unsigned                      _depth;                  //!< Level (depth) of the task
         unsigned char                 _numDevices;             //!< Number of suported devices for this workdescriptor
         DeviceData                  **_devices;                //!< Supported devices for this workdescriptor
         unsigned char                 _activeDeviceIdx;        //!< In _devices, index where we can find the current active DeviceData (if any)
#ifdef GPU_DEV
         int                           _cudaStreamIdx;          //!< FIXME: Only used in CUDA tasks, should not be here...
#endif
         size_t                        _numCopies;              //!< Copy-in / Copy-out data
         CopyData                     *_copies;                 //!< Copy-in / Copy-out data
         size_t                        _paramsSize;             //!< Total size of WD's parameters
         unsigned long                 _versionGroupId;         //!< The way to link different implementations of a task into the same group
         double                        _executionTime;          //!< FIXME:scheduler data. WD starting wall-clock time, accounting data transfers
         double                        _estimatedExecTime;      //!< FIXME:scheduler data. WD estimated execution time, accounting data transfers
         DOSubmit                     *_doSubmit;               //!< DependableObject representing this WD in its parent's depsendencies domain
         LazyInit<DOWait>              _doWait;                 //!< DependableObject used by this task to wait on dependencies
         DependenciesDomain           *_depsDomain;             //!< Dependences domain. Each WD has one where DependableObjects can be submitted            //!< Directory to mantain cache coherence
         nanos_translate_args_t        _translateArgs;          //!< Translates the addresses in _data to the ones obtained by get_address()
         PriorityType                  _priority;               //!< Task priority
         CommutativeOwnerMap          *_commutativeOwnerMap;    //!< Map from commutative target address to owner pointer
         WorkDescriptorPtrList        *_commutativeOwners;      //!< Array of commutative target owners
         int                           _numaNode;               //!< FIXME:scheduler data. The NUMA node this WD was assigned to
         bool                          _copiesNotInChunk;       //!< States whether the buffer of the copies is allocated in the chunk of the WD
         const char                   *_description;            //!< WorkDescriptor description, usually user function name
         InstrumentationContextData    _instrumentationContextData; //!< Instrumentation Context Data (empty if no instr. enabled)
         Slicer                       *_slicer;                 //! Related slicer (NULL if does'nt apply)
         task_reduction_vector_t       _taskReductions;         //< Vector of task reductions
         int                           _criticality;
         //Atomic< std::list<GraphEntry *> * > _myGraphRepList;
         //bool _listed;
         void                        (*_notifyCopy)( WD &wd, BaseThread const &thread);
         BaseThread const             *_notifyThread;
         void const                   *_remoteAddr;
         void                         *_callback;
         void                         *_arguments;
         std::vector<WorkDescriptor *>*_submittedWDs;
         bool                          _reachedTaskwait;
      public:
         int                           _schedValues[8];
         std::map<memory_space_id_t,unsigned int>   _schedPredecessorLocs;
         MemController                 _mcontrol;
      private: /* private methods */
         /*! \brief WorkDescriptor copy assignment operator (private)
          */
         const WorkDescriptor & operator= ( const WorkDescriptor &wd );
         /*! \brief WorkDescriptor default constructor (private)
          */
         WorkDescriptor ();

         //! \brief Adding current WD as descendant of parent (private method)
         void addToGroup ( WorkDescriptor &parent );
      public: /* public methods */
         /*! \brief WorkDescriptor constructor - 1
          */
         WorkDescriptor ( int ndevices, DeviceData **devs, size_t data_size = 0, size_t data_align = 1, void *wdata=0,
                          size_t numCopies = 0, CopyData *copies = NULL, nanos_translate_args_t translate_args = NULL, const char *description = NULL );

         /*! \brief WorkDescriptor constructor - 2
          */
         WorkDescriptor ( DeviceData *device, size_t data_size = 0, size_t data_align = 1, void *wdata=0,
                          size_t numCopies = 0, CopyData *copies = NULL, nanos_translate_args_t translate_args = NULL, const char *description = NULL );

         /*! \brief WorkDescriptor copy constructor (using a given WorkDescriptor)
          *
          *  This function is used as a constructor, receiving as a parameter other WorkDescriptor.
          *  The constructor uses a DeviceData vector and a new void * data which will be completely
          *  different from the former WorkDescriptor. Rest of the data is copied from the former WD.
          *
          *  This constructor is used only for duplicating purposes
          *
          *  \see WorkDescriptor System::duplicateWD
          */
         WorkDescriptor ( const WorkDescriptor &wd, DeviceData **devs, CopyData * copies, void *data = NULL, const char *description = NULL );

         /*! \brief WorkDescriptor destructor
          *
          * All data will be allocated in a single chunk so only the destructors need to be invoked
          * but not the allocator
          */
         virtual ~WorkDescriptor();

         int getId() const { return _id; }
         int getHostId() const { return _hostId; }
         void setHostId( int id ) { _hostId = id; }
         /*! \brief Has this WorkDescriptor ever run?
          */
         bool started ( void ) const;
         bool initialized ( void ) const;

         /*! \brief Prepare WorkDescriptor to run
          *
          *  This function is useful to perform lazy initialization in the workdescriptor
          */
         void init ();
         void initWithPE ( ProcessingElement &pe );

         /*! \brief Last operations just before WD execution
          *
          *  This function is useful to perform any operation that needs to be done at the last moment
          *  before the execution of the WD.
          */
         void start ( ULTFlag isUserLevelThread, WorkDescriptor *previous = NULL );
         void preStart ( ULTFlag isUserLevelThread, WorkDescriptor *previous = NULL );
         bool isInputDataReady();

         /*! \brief Get data size
          *
          *  This function returns the size of the user's data related with current WD
          *
          *  \return data size
          *  \see getData setData setDatasize
          */
         size_t getDataSize () const;

         /*! \brief Get data alignment
          *
          *  This function returns the data alignment of the user's data related with current WD
          *
          *  \return data alignment
          *  \see getData setData setDatasize
          */
         size_t getDataAlignment () const;

         WorkDescriptor * getParent() const;

         void forceParent ( WorkDescriptor * p );

         WDPool * getMyQueue();

         void setMyQueue ( WDPool * myQ );

         bool isEnqueued();

         /*! \brief FIXME: (#170) documentation needed
          *
          *  Named arguments idiom format.
          */
         WorkDescriptor & tied ();

         WorkDescriptor & tieTo ( BaseThread &pe );

         WorkDescriptor & tieToLocation ( memory_space_id_t loc );

         bool isTied() const;

         bool isTiedLocation() const;

         BaseThread * isTiedTo() const;

         memory_space_id_t isTiedToLocation() const;

         bool shouldBeTied() const;

         void untie();

         void untieLocation();

         void setData ( void *wdata );

         void * getData () const;

         void setTotalSize ( size_t size );

         void setBlocked ();

         bool isReady () const;

         void setReady ();

         bool isFinal () const;

         void setFinal ( bool value = true );

         GenericSyncCond * getSyncCond();

         void setSyncCond( GenericSyncCond * syncCond );

         void setDepth ( int l );

         unsigned getDepth() const;

         /* device related methods */
         bool canRunIn ( const Device &device ) const;
         bool canRunIn ( const ProcessingElement &pe ) const;
         DeviceData & activateDevice ( const Device &device );
         DeviceData & activateDevice ( unsigned int deviceIdx );
         DeviceData & getActiveDevice () const;

         bool hasActiveDevice() const;

         void setActiveDeviceIdx( unsigned char idx );
         unsigned char getActiveDeviceIdx() const;

#ifdef GPU_DEV
         void setCudaStreamIdx( int idx );
         int getCudaStreamIdx() const;
#endif

         /*! \brief Sets specific internal data of the programming model
          * \param [in] data Pointer to internal data
          * \param [in] ownedByWD States if the pointer to internal data will be owned by this WD.
          *             If so, it means that it will be deallocated when the WD is destroyed
          */
         void setInternalData ( void *data, bool ownedByWD = true );

         void * getInternalData () const;

         /*! \brief Sets custom data for the scheduling policy
          *  \param [in] data Pointer do the data. Ownership will be
          *  changed to the WD, so that data will be destroyed with it
          *  \param [in] ownedByWD States if the pointer to scheduler
          *  data will be owned by this WD.
          *  If so, it means that it will be deallocated when the WD is
          *  destroyed
          */
         void setSchedulerData( ScheduleWDData * data, bool ownedByWD = true );

         ScheduleWDData* getSchedulerData() const;

         void setTranslateArgs( nanos_translate_args_t translateArgs );

         nanos_translate_args_t getTranslateArgs() const;

         /*! \brief Returns the NUMA node that this WD was assigned to.
          *
          * \see NUMANodet
          */
         int getNUMANode() const;

         /*! \brief Changes the NUMA node this WD is assigned to.
          *
          * \see getNUMANode
          */
         void setNUMANode( int node );

         /*! \brief Get the number of devices
          *
          *  This function return the number of devices for the current WD
          *
          *  \return WorkDescriptor's number of devices
          *  \see getDevices
          */
         unsigned getNumDevices ( void ) const;

         /*! \brief Get devices
          *
          *  This function return a device vector which are related with the current WD
          *
          *  \return devices vector
          *  \see getNumDevices
          */
         DeviceData ** getDevices ( void ) const;

         /*! \brief Prepare device
          *
          *  This function chooses a device from the WD's device list that will run the current WD
          *
          *  \see getDevices
          */
         void prepareDevice ( void );

         /*! \brief WD dequeue
          *
          *  This function give us the next WD slice to execute. As a default
          *  behaviour give the whole WD and returns true, meaning that there
          *  are no more slices to compute
          *
          *  \param [out] slice is the next slice to manage
          *
          *  \return true if there are no more slices to manage, false otherwise
          */
         bool dequeue ( WorkDescriptor **slice );

         // headers
         void submit ( bool force_queue = false );

         bool isOutputDataReady();

         void finish ();
         void preFinish ();

         void done ();

         void clear ();

         /*! \brief returns the number of CopyData elements in the WorkDescriptor
          */
         size_t getNumCopies() const;

         /*! \brief returns the CopyData vector that describes the copy-ins/copy-outs of the WD
          */
         CopyData * getCopies() const;

         /*! \brief returns the total size of copy-ins/copy-outs of the WD
          */
         size_t getCopiesSize() const;

         /*! \brief perform (submit) all output copies
          */
         void submitOutputCopies ();

         /*! \brief wait for output copies to finish
          */
         void waitOutputCopies();

         /*! \brief returns the total size of the WD's parameters
          */
         size_t getParamsSize() const;

         /*! \brief returns the WD's implementation group ID
          */
         unsigned long getVersionGroupId( void );

         /*! \brief sets the WD's implementation group ID
          */
         void setVersionGroupId( unsigned long id );

         /*! \brief returns the total execution time of the WD, accounting data transfers
          */
         double getExecutionTime() const;

         /*! \brief returns the estimated execution time of the WD, accounting data transfers
          */
         double getEstimatedExecutionTime() const;

         /*! \brief sets the estimated execution time of the WD, accounting data transfers
          */
         void setEstimatedExecutionTime( double time );

         /*! \brief Returns a pointer to the DOSubmit of the WD
          */
         DOSubmit * getDOSubmit();

         /*! \brief Returns DOSubmit's number of predecessors
          */
         int getNumDepsPredecessors();

         /*! \brief Returns if DOSubmit's has predecessors
          */
         bool hasDepsPredecessors();

         /*! \brief Add a new WD to the domain of this WD.
          *  \param wd Must be a WD created by "this". wd will be submitted to the
          *  scheduler when its dependencies are satisfied.
          *  \param numDeps Number of dependencies.
          *  \param deps Array with dependencies associated to the submitted wd.
          */
         void submitWithDependencies( WorkDescriptor &wd, size_t numDeps, DataAccess* deps );

         /*! \brief Waits untill all (input) dependencies passed are satisfied for the _doWait object.
          *  \param numDeps Number of de dependencies.
          *  \param deps dependencies to wait on, should be input dependencies.
          */
         void waitOn( size_t numDeps, DataAccess* deps );

         /*! If this WorkDescriptor has an immediate successor (i.e., another WD that only depends on him)
             remove it from the dependence graph and return it. */
         WorkDescriptor * getImmediateSuccessor ( BaseThread &thread );

         /*! \brief Make this WD's domain know a WD has finished.
          *  \paran wd Must be a wd created in this WD's context.
          */
         void workFinished(WorkDescriptor &wd);

         /*! \brief Early-release all the input dependencies of this WD
          */
         void releaseInputDependencies();

         /*! \brief Returns the DependenciesDomain object.
          */
         DependenciesDomain & getDependenciesDomain();

         /*! \brief Returns embeded instrumentation context data.
          */
         InstrumentationContextData *getInstrumentationContextData( void );

         /*! \breif Prepare private copies to have relative addresses
          */
         void prepareCopies();

         //! \brief Wait for all children (1st level work descriptors)
         void waitCompletion( bool avoidFlush = false );

         bool isSubmitted( void ) const;
         void submitted( void );
         bool canBeBlocked( void );

         void notifyOutlinedCompletion();

         void predecessorFinished( WorkDescriptor *predecessorWd );

         void wgdone();
         void listed();
         void printCopies();

         bool isConfigured ( void ) const;
         void setConfigured ( bool value=true );

         void setPriority( PriorityType priority );
         PriorityType getPriority() const;
         void setNotifyCopyFunc( void (*func)(WD &, BaseThread const &) );

         void notifyCopy();

         /*! \brief Store addresses of commutative targets in hash and in child WorkDescriptor.
          *  Called when a task is submitted.
          */
         void initCommutativeAccesses( WorkDescriptor &wd, size_t numDeps, DataAccess* deps );
         /*! \brief Try to take ownership of all commutative targets for exclusive access.
          *  Called when a task is invoked.
          */
         bool tryAcquireCommutativeAccesses();
         /*! \brief Release ownership of commutative targets.
          *  Called when a task is finished.
          */
         void releaseCommutativeAccesses();

         void setImplicit( bool b = true );
         bool isImplicit( void );

         void setRuntimeTask( bool b = true );
         bool isRuntimeTask( void ) const;

         /*! \brief Set copies for a given WD
          * We call this when copies cannot be set at creation time of the work descriptor
          * Note that this should only be done between creation and submit.
          * This function shall not be called if the workdescriptor already has copies.
          *
          * \param numCopies the number of copies. If zero \a copies must be NULL
          * \param copies Buffer of copy descriptors. The workdescriptor WILL NOT acquire the ownership of the copy as a private buffer
          * will be allocated instead
          */
         void setCopies(size_t numCopies, CopyData * copies);

         const char * getDescription ( void ) const;

         //! \brief Removing work from current WorkDescriptor
         virtual void exitWork ( WorkDescriptor &work );

         //! \brief Adding work to current WorkDescriptor
         void addWork( WorkDescriptor &work );

         //! \brief Get related slicer
         Slicer * getSlicer ( void ) const;

         //! \brief Set related slicer
         void setSlicer ( Slicer *slicer );

         //! \brief Convert a slicible WD to a regular WD (changing the behaviour)
         //!
         //! This functions change slicible WD attribute which is used in
         //! submit() and dequeue() when _slicer attribute is specified.
         void convertToRegularWD();

         //! \brief This function registers a new task reduction over a
         //variable if it is not already registered.
         void registerTaskReduction( void *p_orig, size_t p_size, size_t elem_size,
                 void (*p_init)( void *, void * ), void (*p_reducer)( void *, void * ) );

         //! \brief This function registers a new fortran task reduction over an
         //array if it is not already registered.
         void registerFortranArrayTaskReduction( void *p_orig, void *dep,
                 size_t array_descriptor_size, void (*p_init)( void *, void * ),
                 void (*p_reducer)( void *, void * ), void (*p_reducer_orig_var)( void *, void * ) );

         void removeAllTaskReductions ( void );

         void * getTaskReductionThreadStorage( void *p_addr, size_t id );

         TaskReduction * getTaskReduction( const void *p_dep );

         void copyReductions(WorkDescriptor *parent);

         bool resourceCheck( BaseThread const &thd, bool considerInvalidations ) const;

         void setId( unsigned int id );

         void setRemoteAddr( void const *addr );
         void const *getRemoteAddr() const;

         /*! \brief Sets a WorkDescriptor to an invalid state or not depending on the flag value.
             If invalid (flag = true) it propagates upwards to the ancestors until
             no more ancestors exist or a recoverable task is found.
             \param A flag that indicates whether this task is being invalidated or not.
             \return A boolean value that indicates if either the task itself is recoverable or a recoverable ancestor was found.
         */
         bool setInvalid ( bool flag );

         //! \brief Returns whether a WorkDescriptor is invalid or not.
         bool isInvalid ( void ) const;

         //! \brief Marks the WorkDescriptor as recoverable. If the execution of this task is invalid, it will try to re-execute.
         void setRecoverable( bool flag );

         //!brief Returns whether a WorkDescriptor is able to re-execute from the beginning if an error is detected.
         bool isRecoverable ( void ) const;

         void setCriticality ( int cr );
         int getCriticality ( void ) const;

         void setCallback ( void *cb );
         void setArguments ( void *a );

         //! \brief Returns the concurrency level of the WD considering
         //         the commutative access map that the caller provides.
         int getConcurrencyLevel( std::map<WD**, WD*> &comm_accesses ) const;
         void addPresubmittedWDs( unsigned int numWDs, WD **wds );
   };

   typedef class WorkDescriptor WD;

   typedef class DeviceData DD;

} // namespace nanos

#endif
