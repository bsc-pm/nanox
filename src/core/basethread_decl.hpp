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

#ifndef _BASE_THREAD_DECL
#define _BASE_THREAD_DECL

#include <set>
#include <fstream>

#include "processingelement_fwd.hpp"
#include "schedule_fwd.hpp"
#include "threadteam_fwd.hpp"

#include "debug.hpp"
#include "atomic_decl.hpp"
#include "lock_decl.hpp"

#include "workdescriptor_decl.hpp"
#include "allocator_decl.hpp"
#include "wddeque_decl.hpp"

namespace nanos {

   typedef void SchedulerHelper ( WD *oldWD, WD *newWD, void *arg); // FIXME: should be only in one place

   /*!
    * Each thread in a team has one of this. All data associated with the team should be here
    * and not in BaseThread as it needs to be saved and restored on team switches
    */
   class TeamData
   {
      typedef ScheduleThreadData SchedData;
      
      private:
         unsigned          _id;
         ThreadTeam       *_team;
         unsigned          _singleCount;
         SchedData        *_schedData;
         TeamData         *_parentData;
         nanos_ws_desc_t  *_wsDescriptor; /*< pointer to last worksharing descriptor */
         bool              _star;         /*< is current thread a star (or role) within the team */
         bool              _creator;      /*< is team creator */

      private:
        /*! \brief TeamData copy constructor (private)
         */
         TeamData ( const TeamData &td );
        /*! \brief TeamData copy assignment operator (private)
         */
         TeamData& operator= ( const TeamData &td );
      public:
        /*! \brief TeamData default constructor
         */
         TeamData () : _id( 0 ), _team(NULL), _singleCount( 0 ), _schedData( NULL ), _parentData( NULL ), _wsDescriptor( NULL ), _star( false ), _creator(false) {}
        /*! \brief TeamData destructor
         */
         ~TeamData ();

         unsigned getId() const { return _id; }
         ThreadTeam *getTeam() const { return _team; }
         unsigned getSingleCount() const { return _singleCount; }

         void setId ( int id ) { _id = id; }
         void setTeam ( ThreadTeam *team ) { _team = team; }
         unsigned nextSingleBarrierGuard() {
            ++(++_singleCount);
            return _singleCount;
         }
         unsigned nextSingleGuard() {
            ++_singleCount;
            return _singleCount;
         }
         unsigned currentSingleGuard() {
            return _singleCount;
         }
        
        /*! \brief Returns if related thread is starring within current team
         */
         bool isStarring ( void ) const ;

        /*! \brief Set related thread as star thread (or not)
         */
         void setStar ( bool v = true ) ;


         void setScheduleData ( SchedData *data ) { _schedData = data; }
         SchedData * getScheduleData () const { return _schedData; }

        /*! \brief Returns next global worksharing descriptor for _team 
         */
         nanos_ws_desc_t *getTeamWorkSharingDescriptor( BaseThread * thread, bool *b );

         void setParentTeamData ( TeamData *data ) { _parentData = data; }
         TeamData * getParentTeamData () const { return _parentData; }

         void setCreator ( bool value ) ;
         bool isCreator ( void ) const;
   };

   namespace ext {
   class SMPMultiThread;
   };

   class BaseThread
   {
      friend class Scheduler;
      private:
         typedef void (*callback_t)(void);
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
             struct AtomicBool
             {
                 private:
                     bool value;
                 public:
                     AtomicBool() : value() { }
                     /* explicit */ operator bool() const {
                         return __atomic_load_n(&value, __ATOMIC_ACQUIRE);
                     };
                     AtomicBool& operator=(bool b)
                     {
                         __atomic_store_n(&value, b, __ATOMIC_RELEASE);
                         return *this;
                     }

                     bool operator!() const
                     {
                         return !this->operator bool();
                     }
             };
#endif
         struct StatusFlags {
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
             typedef AtomicBool StatusFlag;
#else
             typedef bool StatusFlag;
#endif
            StatusFlag is_main_thread;
            StatusFlag has_started;
            StatusFlag must_stop;
            StatusFlag must_sleep;
            StatusFlag is_idle;
            StatusFlag is_paused;
            StatusFlag has_team;
            StatusFlag has_joined;
            StatusFlag is_waiting;
            StatusFlag can_get_work;    /**< Set whether the thread can get more WDs to run or not */
            StatusFlag must_leave_team; /**< Set whether to leave the team when thread is blocked */

            StatusFlags()
                : is_main_thread(), has_started(), must_stop(),
                must_sleep(), is_idle(), is_paused(), has_team(), has_joined(),
                is_waiting(), can_get_work()
             { }
         };
      private:
         // Thread info/status
         unsigned short          _id;            /**< Thread identifier */
         unsigned int            _osId;          /**< OS Thread identifier */
         unsigned short          _maxPrefetch;   /**< Maximum number of tasks that the thread can be running simultaneously */
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
         StatusFlags    _status;        /**< BaseThread status flags */
#else
         volatile StatusFlags    _status;        /**< BaseThread status flags */
#endif
         ext::SMPMultiThread    *_parent;
         // Relationships:
         ProcessingElement      *_pe;            /**< Threads are binded to a PE for its life-time */
         // Thread synchro:
         Lock                    _mlock;         /**< Thread Lock */
         // Current/following tasks: 
         WD                     &_threadWD;      /**< Thread implicit WorkDescriptor */
         WD                     *_currentWD;     /**< Current WorkDescriptor the thread is executing */
         WD                     *_heldWD;
         WDDeque                 _nextWDs;       /**< Queue with all the tasks that the thread is being run simultaneously */
         // Thread's Team info:
         TeamData               *_teamData;      /**< Current team data, thread is registered and also it has entered to the team */
         TeamData               *_nextTeamData;  /**< Next team data, thread is already registered in a new team but has not enter yet */
         nanos_ws_desc_t         _wsDescriptor;  /**< Local worksharing descriptor */
         // Name and description:
         std::string             _name;          /**< Thread name */
         std::string             _description;   /**< Thread description */
         // Allocator:
         Allocator               _allocator;     /**< Per thread allocator */
         unsigned short          _steps;         //!< Number of scheduler steps (zero means infinite)
         callback_t              _bpCallBack;    //!< Break point callback. We call it after _steps scheduler ops
         ThreadTeam             *_nextTeam;      //!< If thread has no team, which team should it join

      private:
         virtual void runDependent () = 0;

         // These must be called through the Scheduler interface
         virtual void switchHelperDependent( WD* oldWD, WD* newWD, void *arg ) = 0;
         virtual void exitHelperDependent( WD* oldWD, WD* newWD, void *arg ) = 0;
         virtual bool inlineWorkDependent (WD &work) = 0;
         virtual void switchTo( WD *work, SchedulerHelper *helper ) = 0;
         virtual void exitTo( WD *work, SchedulerHelper *helper ) = 0;

      private:
         //! \brief BaseThread default constructor (private)
         BaseThread ();
         //! \brief BaseThread copy constructor (private)
         BaseThread( const BaseThread & );
         //! \brief BaseThread copy assignment operator (private)
         const BaseThread & operator= ( const BaseThread & );
      public:
         std::ostream          *_file;
         bool                   _gasnetAllowAM;
         std::set<void *> _pendingRequests;
         //! \brief BaseThread constructor
         BaseThread ( unsigned int osId, WD &wd, ProcessingElement *creator = 0, ext::SMPMultiThread *parent = NULL );

         //! \brief BaseThread destructor
         virtual ~BaseThread()
         {
            finish();
            ensure0( ( !_status.has_team ), "Destroying thread inside a team!" );
            ensure0( ( _status.has_joined || _status.is_main_thread ), "Trying to destroy running thread" );
         }

         // atomic access
         virtual void lock ();
         virtual void unlock ();

         virtual void initializeDependent () = 0;
         virtual void start () = 0;
         virtual void finish ();
         void run();
         void stop();
         virtual void sleep();
         virtual void wakeup();

         void pause ();
         void unpause ();

         virtual void idle( bool debug = false ) {};
         virtual void processTransfers();
         virtual void yield() {};

         /*! \brief Called when the thread becomes blocked inside a SynchronizedCondition and it
                    cannot execute any task (neither implicit thread task)
          */
         virtual void atBlock() { idle(); };

         /*! \brief Must be called by children classes after the join operation (protected)
          */ 
         void joined ( void ); 
         virtual void join() = 0;

         bool hasJoined() const;

         virtual void bind() {};
         virtual void outlineWorkDependent (WD &work) = 0;
         virtual void preOutlineWorkDependent (WD &work) = 0;

         virtual void wait();
         virtual void resume();

         virtual bool canBlock() { return false; }

         // set/get methods
         void setHeldWD ( WD *wd );
         WD * getHeldWD () const;
         void setCurrentWD ( WD &current );

         WD * getCurrentWD () const;

         WD & getThreadWD () const;

         // Prefetching related methods used also by slicers
         int getMaxPrefetch () const;
         void setMaxPrefetch ( int max );
         bool canPrefetch () const;
         virtual void addNextWD ( WD *next );
         virtual WD * getNextWD ();
         virtual bool hasNextWD () const;

         // Return the number of concurrent tasks (tasks that can be run by this thread at the same time)
         int getMaxConcurrentTasks() const;

         // Set whether the thread will schedule WDs or not used by getImmediateSuccessor()
         // If so, WD's dependencies should be kept till WD is finished
         virtual bool keepWDDeps() { return false; }

         ext::SMPMultiThread *getParent() ;
         virtual BaseThread *getNextThread() = 0;

         // team related methods
         void reserve();
         void enterTeam( TeamData *data = NULL ); 
         bool hasTeam() const;
         void leaveTeam();

         ThreadTeam * getTeam() const;
         TeamData * getTeamData() const;
         TeamData * getNextTeamData() const;

         void setNextTeamData( TeamData *td );

        /*! \brief Returns the address of the local worksharing descriptor
         */
         nanos_ws_desc_t *getLocalWorkSharingDescriptor( void );

        /*! \brief Returns the address of a global worksharing descriptor (provided for team, through team data)
         */
         nanos_ws_desc_t *getTeamWorkSharingDescriptor( bool *b );

         //! Returns the id of the thread inside its current team 
         int getTeamId() const;

         bool isStarted () const;

         bool isRunning () const;

         bool isSleeping () const;
         
         bool isTeamCreator () const;

         //! \brief Is the thread paused as the result of stopping the scheduler?
         bool isPaused () const;

         virtual bool canGetWork ();

         void enableGettingWork ();

         void disableGettingWork ();

         bool isLeavingTeam () const;

         void setLeaveTeam ( bool leave );

         ProcessingElement * runningOn() const;
         
         void setRunningOn(ProcessingElement* element);
         
         void associate( WD *wd = NULL );

         int getId() const;

         virtual int getCpuId() const;

         bool singleGuard();
         bool enterSingleBarrierGuard ();
         void releaseSingleBarrierGuard ();
         void waitSingleBarrierGuard ();

        /*! \brief Returns if related thread is starring within team 't'
         */
         bool isStarring ( const ThreadTeam * t ) const ;

        /*! \brief Set related thread as star thread (or not)
         */
         void setStar ( bool v = true ) ;

         /*! \brief Get allocator for current thread 
          */
         Allocator & getAllocator();

         /*! \brief Rename the basethread
          */
         void rename ( const char *name );

         /*! \brief Get BaseThread name
          */
         const std::string &getName ( void ) const;

         /*! \brief Get BaseThread description
          */
         const std::string &getDescription ( void );

         virtual void switchToNextThread() = 0;
         virtual void notifyOutlinedCompletionDependent( WD *completedWD );
         virtual bool isCluster() = 0;
         WDDeque &getNextWDQueue();

         /*! \brief Get Status: Main Thread
          */
         bool isMainThread ( void ) const;
         /*! \brief Get Status: Is waiting
          */
         bool isWaiting () const; 
         /*! \brief Set Status: Main Thread
          */
         void setMainThread ( bool v = true );

#ifdef NANOS_RESILIENCY_ENABLED
         /*! \brief Change the action taken by default if some specified signals are received.
          */
         virtual void setupSignalHandlers() = 0;

#endif
         //! \brief Wake up a thread
         void tryWakeUp( ThreadTeam *team );

         unsigned int getOsId() const;

         //! \brief Change thread state idle to value ( true by default )
         void setIdle ( bool value = true ) ;
         //! \brief Inquiry thread state idle
         bool isIdle ( void ) const;
         //! \brief Basethread step.
         void step( void );
         //! \brief Set break point steps 
         void setSteps( unsigned short s );
         //! \brief Set break point callback
         void setCallBack( callback_t cb );

         //! \brief Get next Team to enter
         ThreadTeam* getNextTeam() const;
         //! \brief Set next Team to enter
         void setNextTeam( ThreadTeam *team );
   };

   extern __thread BaseThread *myThread;

   BaseThread * getMyThreadSafe();

} // namespace nanos

#endif
