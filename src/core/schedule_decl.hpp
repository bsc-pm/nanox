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

#ifndef _NANOS_SCHEDULE_DECL_H
#define _NANOS_SCHEDULE_DECL_H

#include <stddef.h>
#include <string>

#include "debug.hpp"

#include "synchronizedcondition_fwd.hpp"
#include "system_fwd.hpp"

#include "workdescriptor_decl.hpp"
#include "atomic_decl.hpp"
#include "functors_decl.hpp"
#include "basethread_decl.hpp"


namespace nanos {

   class Config; // FIXME: this should be on config_fwd

// singleton class to encapsulate scheduling data and methods
   typedef void SchedulerHelper ( WD *oldWD, WD *newWD, void *arg);

   class Scheduler
   {
      private:
         static void switchHelper (WD *oldWD, WD *newWD, void *arg);
         static void exitHelper (WD *oldWD, WD *newWD, void *arg);

         template<class behaviour>
         static void idleLoop (void);

      public:
         static bool tryPreOutlineWork ( WD *work );
         static void preOutlineWork ( WD *work );
         static void prePreOutlineWork ( WD *work );
         static void preOutlineWorkWithThread ( BaseThread *thread, WD *work );
         static void postOutlineWork ( WD *work, bool schedule, BaseThread *owner );
         static bool inlineWork ( WD *work, bool schedule );
         static bool inlineWorkAsync ( WD *wd, bool schedule );
         static void outlineWork( BaseThread *currentThread, WD *wd );

         static void submit ( WD &wd, bool force_queue = false );
         static void _submit ( WD &wd, bool force_queue = false );
         /*! \brief Submits a set of wds. It only calls the policy's queue()
          *  method!
          */
         static void submit ( WD ** wds, size_t numElems );
         static void _submit ( WD ** wds, size_t numElems );
         static void switchTo ( WD *to );
         static void exitTo ( WD *next );
         static void switchToThread ( BaseThread * thread );
         static void finishWork( WD * wd, bool schedule );

         static void workerLoop ( void );
         static void asyncWorkerLoop ( void );
         static void yield ( void );

         static void exit ( void );

         static void waitOnCondition ( GenericSyncCond *condition );
         static void wakeUp ( WD *wd );

         static WD * prefetch ( BaseThread *thread, WD &wd );

         static void updateExitStats ( WD &wd );
         static void updateCreateStats ( WD &wd );

         /*! \brief checks if a WD is elegible to run in a given thread */
         static bool checkBasicConstraints ( WD &wd, BaseThread const &thread );
   };

   class SchedulerConf
   {
      friend class System;
      private: /* PRIVATE DATA MEMBERS */
         unsigned int                  _numSpins;          //!< Number of spins before yield
         unsigned int                  _numChecks;         //!< Number of checks before schedule
         bool                          _schedulerEnabled;  //!< Scheduler is enabled
         int                           _numStealAfterSpins;//!< Steal every so spins
         bool                          _holdTasks;         //!< Submit tasks when a taskwait is reached
      private: /* PRIVATE METHODS */
        //! \brief SchedulerConf default constructor (private)
        SchedulerConf() : _numSpins(1), _numChecks(1), _schedulerEnabled(true),
        _numStealAfterSpins(1), _holdTasks(false) {}
        //! \brief SchedulerConf copy constructor (private)
        SchedulerConf ( SchedulerConf &sc ) : _numSpins(), _numChecks(),
        _schedulerEnabled(), _holdTasks()
        {
           fatal("SchedulerConf: Illegal use of class");
        }
        //! \brief SchedulerConf copy assignment operator (private)
        SchedulerConf & operator= ( SchedulerConf &sc );
      public: /* PUBLIC METHODS */
         //! \brief SchedulerConf destructor
         ~SchedulerConf() {}

         //! \brief Set if scheduler is enabled
         void setSchedulerEnabled ( const bool value ) ;

         //! \brief Returns the number of spins before yield
         unsigned int getNumSpins ( void ) const;
         //! \brief Returns the number of checks before schedule
         unsigned int getNumChecks ( void ) const;
         //! \brief Returns the number of spins before stealing
         unsigned int getNumStealAfterSpins ( void ) const;
         //! \brief Returns if scheduler is enabled
         bool getSchedulerEnabled () const;
         //! \brief Returns if holding tasks is enabled
         bool getHoldTasksEnabled () const;

         //! \brief Configure scheduler runtime options
         void config ( Config &cfg );
   };

   class SchedulerStats
   {
         friend class WDDeque;
         friend class WDLFQueue;
         friend class WDPriorityQueue<WD::PriorityType>;
         friend class WDPriorityQueue<double>;
         friend class Scheduler;
         friend class System;

         friend class SlicerStaticFor;
         friend class SlicerDynamicFor;
         friend class SlicerGuidedFor;
         friend class SlicerRepeatN;
         friend class SlicerCompoundWD;
      private:
         Atomic<int>          _createdTasks;
         Atomic<int>          _readyTasks;
         Atomic<int>          _idleThreads;
         Atomic<int>          _totalTasks;
      private:
         /*! \brief SchedulerStats copy constructor (private)
          */
         SchedulerStats ( SchedulerStats &ss );
         /*! \brief SchedulerStats copy assignment operator (private)
          */
         SchedulerStats & operator= ( SchedulerStats &ss );
      public:
         /*! \brief SchedulerStats default constructor
          */
         SchedulerStats () : _createdTasks(0), _readyTasks(0), _idleThreads(0), _totalTasks(1) {}
         /*! \brief SchedulerStats destructor
          */
         ~SchedulerStats () {}

         int getCreatedTasks();
         int getReadyTasks();
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
         int * getReadyTasksAddr( void ) { return &_readyTasks.override(); }
#else
         volatile int * getReadyTasksAddr( void ) { return &_readyTasks.override(); }
#endif
         int getTotalTasks();
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
         int * getTotalTasksAddr( void ) { return &_totalTasks.override(); }
#else
         volatile int * getTotalTasksAddr( void ) { return &_totalTasks.override(); }
#endif
   };

   class ScheduleTeamData {
      private:
         /*! \brief ScheduleTeamData copy constructor (private)
          */
         ScheduleTeamData ( ScheduleTeamData &std );
         /*! \brief ScheduleTeamData copy assignment operator (private)
          */
         ScheduleTeamData& operator=  ( ScheduleTeamData &std );
      public:
         /*! \brief ScheduleTeamData default constructor
          */
         ScheduleTeamData() {}
         /*! \brief ScheduleTeamData destructor
          */
         virtual ~ScheduleTeamData() {}

         /*! \brief Print the statistics of the ScheduleTeamData, if any
          */
         virtual void printStats() {}
   };

   class ScheduleThreadData {
      private:
         /*! \brief ScheduleThreadData copy constructor (private)
          */
         ScheduleThreadData( ScheduleThreadData &std );
         /*! \brief ScheduleThreadData copy assignment operator (private)
          */
         ScheduleThreadData& operator= ( ScheduleThreadData &std );
      public:
         /*! \brief ScheduleThreadData default constructor
          */
         ScheduleThreadData() {}
         /*! \brief ScheduleThreadData destructor
          */
         virtual ~ScheduleThreadData() {}
   };

   class ScheduleWDData {
      private:
         /*! \brief ScheduleWDData copy constructor (private)
          */
         ScheduleWDData( ScheduleWDData &std );
         /*! \brief ScheduleWDData copy assignment operator (private)
          */
         ScheduleWDData& operator= ( ScheduleWDData &std );
      public:
         /*! \brief ScheduleWDData default constructor
          */
         ScheduleWDData() {}
         /*! \brief ScheduleWDData destructor
          */
         virtual ~ScheduleWDData() {}
   };

   class SchedulePolicy
   {
      public:
         typedef enum {
            SYS_SUBMIT, SYS_SUBMIT_WITH_DEPENDENCIES, SYS_INLINE_WORK
         } SystemSubmitFlag;

      private:
         std::string    _name;
      private:
         /*! \brief SchedulePolicy default constructor (private)
          */
         SchedulePolicy ();
         /*! \brief SchedulePolicy copy constructor (private)
          */
         SchedulePolicy ( SchedulePolicy &sp );
         /*! \brief SchedulePolicy copy assignment operator (private)
          */
         SchedulePolicy& operator= ( SchedulePolicy &sp );
      public:
         /*! \brief SchedulePolicy constructor - with std::string &name
          */
         SchedulePolicy ( std::string &name ) : _name(name) {}
         /*! \brief SchedulePolicy constructor - with char *name
          */
         SchedulePolicy ( const char *name ) : _name(name) {}
         /*! \brief SchedulePolicy destructor
          */
         virtual ~SchedulePolicy () {};

         const std::string & getName () const;

         virtual size_t getTeamDataSize() const = 0;
         virtual size_t getThreadDataSize() const = 0;
         virtual ScheduleTeamData * createTeamData () = 0;
         virtual ScheduleThreadData * createThreadData () = 0;

         virtual size_t getWDDataSize () const { return 0; }
         virtual size_t getWDDataAlignment () const { return 0; }
         virtual void initWDData ( void * data ) const {}

         virtual WD * atSubmit      ( BaseThread *thread, WD &wd ) = 0;

         /*! \brief \param numSteal The core scheduler will set this parameter
          *  0 if no steal operation should be allowed.
          *  Otherwise, attempt a steal operation instead of a normal
          *  atIdle op.
          *  This parameter indicates the number of steal attempts
          *  during the current spin loop. The policy might use this
          *  information to progressively relax the stealing conditions
          */
         virtual WD * atIdle        ( BaseThread *thread, int numSteal ) = 0;

         /*! \brief \param schedule The core scheduler will set this parameter
          *  0 if no WD should be returned.
          *  Otherwise, a WD can be returned and it will be queued to local thread queue
          */
         virtual WD * atBeforeExit  ( BaseThread *thread, WD &current, bool schedule );

         /*! \brief \see atIdle for an explanation of the numSteal
          *  parameter
          */
         virtual WD * atAfterExit   ( BaseThread *thread, WD *current, int numSteal );
         virtual WD * atBlock       ( BaseThread *thread, WD *current );
         virtual WD * atYield       ( BaseThread *thread, WD *current);
         virtual WD * atWakeUp      ( BaseThread *thread, WD &wd );
         virtual WD * atPrefetch    ( BaseThread *thread, WD &current );
         virtual void atCreate      ( DependableObject &depObj );
         virtual void atSupport     ( BaseThread *thread );
         virtual void atShutdown    ( void );
         virtual void atSuccessor   ( DependableObject &depObj, DependableObject &pred );

         virtual void queue ( BaseThread *thread, WD &wd )  = 0;
         /*! \brief Batch processing version.
          *  The default behaviour calls queue() individually.
          */
         virtual void queue ( BaseThread ** threads, WD ** wds, size_t numElems );

         /*! \brief Checks if the WD can be batch processed by this policy.
          *  By default returns always false.
          */
         virtual bool isValidForBatch ( const WD * wd ) const { return false; }

         /*! \brief Hook function called when a WD is submitted.
          \param wd [in] The WD to be submitted.
          \param from [in] A flag indicating where the method is called from.
          \sa SystemSubmitFlag.
          */
         virtual void onSystemSubmit( const WD &wd, SystemSubmitFlag from ) {}

         /*! \brief This method will be called when a pair of preceeding and
          succeeding work descriptors is found.
          \param predecessor Preceeding Dependable Object pointer.
          \param successor ...
          */
         virtual void successorFound( DependableObject *predecessor, DependableObject *successor ) {}

         /*! \brief Enables or disables stealing */
         virtual void setStealing( bool value ) {}

         /*! \brief Returns the status of stealing */
         virtual bool getStealing()
         {
            return false;
         }
         //! \brief Partial reorder on WD's priority queue
         virtual bool reorderWD( BaseThread *t, WD * wd )
         {
            return true;
         }

         /*! \brief Returns true if there's some WD that can be dequeued.
          */
         virtual bool testDequeue();

         /*! \brief Returns if the scheduler needs WD execution time */
         virtual bool isCheckingWDExecTime()
         {
            return false;
         }

         /*! \brief Returns if priorities are enabled in this policy.
          * Note that some policies won't support priorities while others
          * will depending on a flag
          */
         virtual bool usingPriorities() const
         {
            return false;
         }

         virtual std::string getSummary() const
         {
            return std::string();
         }
   };
   /*! \brief Functor that will be used when a WD's predecessor is found.
    */
   struct SchedulePolicySuccessorFunctor
   {
      SchedulePolicy& _obj;

      SchedulePolicySuccessorFunctor( SchedulePolicy& obj ) : _obj( obj ) {}

      void operator() ( DependableObject *predecessor, DependableObject *successor );
   };

} // namespace nanos

#endif
