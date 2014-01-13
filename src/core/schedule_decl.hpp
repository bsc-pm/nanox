
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

         static void submit ( WD &wd );
         static void submitAndWait ( WD &wd );
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

      private:
        unsigned int  _numSpins;
        int  _numSleeps;
        int  _timeSleep;
        bool _schedulerEnabled;
      private:
        /*! \brief SchedulerConf default constructor (private)
         */
        SchedulerConf() : _numSpins(100), _numSleeps(20), _timeSleep(100), _schedulerEnabled(true) {}
        /*! \brief SchedulerConf copy constructor (private)
         */
        SchedulerConf ( SchedulerConf &sc ) : _numSpins( sc._numSpins ), _numSleeps(sc._numSleeps), _timeSleep(sc._timeSleep), _schedulerEnabled( true )  {}
        /*! \brief SchedulerConf copy assignment operator (private)
         */
        SchedulerConf & operator= ( SchedulerConf &sc );

      public:
         /*! \brief SchedulerConf destructor 
          */
         ~SchedulerConf() {}

         unsigned int getNumSpins () const;
         void setNumSpins ( const unsigned int num );
         int getNumSleeps () const;
         void setNumSleeps ( const unsigned int num );
         int getTimeSleep () const;
         void setSchedulerEnabled ( bool value ) ;
         bool getSchedulerEnabled () const;
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

