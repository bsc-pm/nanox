
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

#ifndef _NANOS_SCHEDULE_DECL_H
#define _NANOS_SCHEDULE_DECL_H

#include <stddef.h>
#include <string>

#include "workdescriptor_decl.hpp"
#include "atomic_decl.hpp"
#include "functors_decl.hpp"
#include "synchronizedcondition_fwd.hpp"
#include "system_fwd.hpp"
#include "basethread_decl.hpp"

namespace nanos
{
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
         static bool inlineWork ( WD *work, bool schedule = false );

         static void submit ( WD &wd, bool force_queue = false  );
         /*! \brief Submits a set of wds. It only calls the policy's queue()
          *  method!
          */
         static void submit ( WD ** wds, size_t numElems  );
         static void switchTo ( WD *to );
         static void exitTo ( WD *next );
         static void switchToThread ( BaseThread * thread );
         static void finishWork( WD * wd, bool schedule = false );

         static void workerLoop ( void );
         static void yield ( void );

         static void exit ( void );

         static void waitOnCondition ( GenericSyncCond *condition );
         static void wakeUp ( WD *wd );

         static WD * prefetch ( BaseThread *thread, WD &wd );

         static void updateExitStats ( WD &wd );
         static void updateCreateStats ( WD &wd );

         /*! \brief checks if a WD is elegible to run in a given thread */
         static bool checkBasicConstraints ( WD &wd, BaseThread &thread );
   };

   class SchedulerConf
   {
      friend class System;
      private: /* PRIVATE DATA MEMBERS */
         unsigned int                  _numSpins;          //!< Number of spins before yield
         unsigned int                  _numChecks;         //!< Number of checks before schedule
         unsigned int                  _numYields;         //!< Number of yields before block
         bool                          _useYield;          //!< Yield is allowed
         bool                          _useBlock;          //!< Block is allowed
         bool                          _schedulerEnabled;  //!< Scheduler is enabled
      private: /* PRIVATE METHODS */
        //! \brief SchedulerConf default constructor (private)
        SchedulerConf() : _numSpins(1), _numChecks(1), _numYields(1), _useYield(false), _useBlock(false), _schedulerEnabled(true) {}
        //! \brief SchedulerConf copy constructor (private)
        SchedulerConf ( SchedulerConf &sc ) : _numSpins(), _numChecks(), _numYields(), _useYield(), _useBlock(), _schedulerEnabled()
        {
           fatal("SchedulerConf: Illegal use of class");
        }
        //! \brief SchedulerConf copy assignment operator (private)
        SchedulerConf & operator= ( SchedulerConf &sc );
      public: /* PUBLIC METHODS */
         //! \brief SchedulerConf destructor 
         ~SchedulerConf() {}

         //! \brief Set if yield is allowed
         void setUseYield ( const bool value );
         //! \brief Set if block is allowed
         void setUseBlock ( const bool value );
         //! \brief Set if scheduler is enabled
         void setSchedulerEnabled ( const bool value ) ;

         //! \brief Returns the number of spins before yield
         unsigned int getNumSpins ( void ) const;
         //! \brief Returns the number of checks before schedule
         unsigned int getNumChecks ( void ) const;
         //! \brief Returns the number of yields before block
         unsigned int getNumYields ( void ) const;
         //! \brief Returns if yield is allowed
         bool getUseYield ( void ) const;
         //! \brief Returns if block is allowed
         bool getUseBlock ( void ) const;
         //! \brief Returns if scheduler is enabled 
         bool getSchedulerEnabled () const;

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
         int getTotalTasks();
         volatile int * getTotalTasksAddr( void ) { return &_totalTasks.override(); }
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
         
         virtual WD * atSubmit      ( BaseThread *thread, WD &wd ) = 0;
         virtual WD * atIdle        ( BaseThread *thread ) = 0;
         virtual WD * atBeforeExit  ( BaseThread *thread, WD &current, bool schedule );
         virtual WD * atAfterExit   ( BaseThread *thread, WD *current );
         virtual WD * atBlock       ( BaseThread *thread, WD *current );
         virtual WD * atYield       ( BaseThread *thread, WD *current);
         virtual WD * atWakeUp      ( BaseThread *thread, WD &wd );
         virtual WD * atPrefetch    ( BaseThread *thread, WD &current );

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

         /*! \brief Returns the number of ready tasks that could be ran simultaneously
          * Tied and commutative WDs in the queue could decrease this number.
          */
         virtual int getPotentiallyParallelWDs( void );
   };
   /*! \brief Functor that will be used when a WD's predecessor is found.
    */
   struct SchedulePolicySuccessorFunctor
   {
      SchedulePolicy& _obj;
      
      SchedulePolicySuccessorFunctor( SchedulePolicy& obj ) : _obj( obj ) {}
      
      void operator() ( DependableObject *predecessor, DependableObject *successor );
   };
   
};

#endif

