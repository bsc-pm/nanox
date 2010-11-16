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

#ifndef _BASE_THREAD_ELEMENT
#define _BASE_THREAD_ELEMENT

#include "workdescriptor.hpp"
#include "atomic.hpp"
#include "processingelement.hpp"
#include "debug.hpp"
#include "schedule_fwd.hpp"
#include "threadteam_fwd.hpp"

namespace nanos
{
   typedef void SchedulerHelper ( WD *oldWD, WD *newWD, void *arg); // FIXME: should be only in one place

   /*!
    * Each thread in a team has one of this. All data associated with the team should be here
    * and not in BaseThread as it needs to be saved and restored on team switches
    */
   class TeamData
   {
      typedef ScheduleThreadData SchedData;
      
      private:
         unsigned       _id;
         unsigned       _singleCount;
         SchedData    * _schedData;
         // PM Data?

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
         TeamData () : _id( 0 ), _singleCount( 0 ), _schedData( NULL ) {}
        /*! \brief TeamData destructor
         */
         ~TeamData () {}

         unsigned getId() const { return _id; }
         unsigned getSingleCount() const { return _singleCount; }

         void setId ( int id ) { _id = id; }
         unsigned nextSingleGuard() {
            ++_singleCount;
            return _singleCount;
         }

         void setScheduleData ( SchedData *data ) { _schedData = data; }
         SchedData * getScheduleData () const { return _schedData; }
   };



   class BaseThread
   {
      friend class Scheduler;

      private:
         static Atomic<int>      _idSeed;
         Lock                    _mlock;

         // Thread info
         int                     _id;
         std::string             _name;
         std::string             _description;

         ProcessingElement *     _pe;         /**< Threads are binded to a PE for its life-time */
         WD &                    _threadWD;

         // Thread status
         bool                    _started;
         volatile bool           _mustStop;
         WD *                    _currentWD;
         WD *                    _nextWD;

         // Team info
         bool                    _hasTeam;
         ThreadTeam *            _team;
         TeamData *              _teamData;
//         int                     _teamId; //! Id of the thread inside its current team
//          int                     _localSingleCount;

         virtual void initializeDependent () = 0;
         virtual void runDependent () = 0;

         // These must be called through the Scheduler interface
         virtual void switchHelperDependent( WD* oldWD, WD* newWD, void *arg ) = 0;
         virtual void exitHelperDependent( WD* oldWD, WD* newWD, void *arg ) = 0;
         virtual void inlineWorkDependent (WD &work) = 0;
         virtual void switchTo( WD *work, SchedulerHelper *helper ) = 0;
         virtual void exitTo( WD *work, SchedulerHelper *helper ) = 0;

      protected:

         /*!
          *  Must be called by children classes after the join operation
          */ 
         void joined ()
         {
            _started = false;
         }

      private:
        /*! \brief BaseThread default constructor
         */
         BaseThread ();
        /*! \brief BaseThread copy constructor (private)
         */
         BaseThread( const BaseThread & );
        /*! \brief BaseThread copy assignment operator (private)
         */
         const BaseThread & operator= ( const BaseThread & );
      public:
        /*! \brief BaseThread constructor
         */
         BaseThread ( WD &wd, ProcessingElement *creator=0 ) :
               _id( _idSeed++ ), _name("Thread"), _description(""), _pe( creator ), _threadWD( wd ),
               _started( false ), _mustStop( false ), _currentWD( NULL),
               _nextWD( NULL), _hasTeam( false ),_team(NULL),
               _teamData(NULL) {}
        /*! \brief BaseThread destructor
         */
         virtual ~BaseThread() {
            ensure0(!_hasTeam,"Destroying thread inside a team!");
            ensure0(!_started,"Trying to destroy running thread");
         }

         // atomic access
         void lock () { _mlock++; }

         void unlock () { _mlock--; }

         virtual void start () = 0;
         void run();
         void stop() { _mustStop = true; }

         virtual void idle() {};
         virtual void yield() {};

         virtual void join() = 0;
         virtual void bind() {};

         // set/get methods
         void setCurrentWD ( WD &current ) { _currentWD = &current; }

         WD * getCurrentWD () const { return _currentWD; }

         WD & getThreadWD () const { return _threadWD; }

         void setNextWD ( WD *next ) { _nextWD = next; }

         WD * getNextWD () const { return _nextWD; }

         // team related methods
         void reserve() { _hasTeam = 1; }

         void enterTeam( ThreadTeam *newTeam, TeamData *data ) {
            _teamData = data;
	    ::memoryFence();
            _team = newTeam;
            _hasTeam=1;
         }

         bool hasTeam() const { return _hasTeam; }

         void leaveTeam() { _hasTeam = 0; _team = 0; }

         ThreadTeam * getTeam() const { return _team; }

         TeamData * getTeamData() const { return _teamData; }

         //! Returns the id of the thread inside its current team 
         int getTeamId() const { return _teamData->getId(); }

         bool isStarted () const { return _started; }

         bool isRunning () const { return _started && !_mustStop; }

         ProcessingElement * runningOn() const { return _pe; }

         void associate();

         int getId() { return _id; }

         int getCpuId() { return runningOn()->getId(); }

         bool singleGuard();

         /*! \brief Rename the basethread
          */
         void rename ( const char *name )
         {
            _name = *new std::string(name);
         }

         /*! \brief Get BaseThread name
          */
         const std::string getName ( void )
         {
            return _name;
         }

         /*! \brief Get BaseThread description
          */
         const std::string getDescription ( void )
         {
            if ( _description.compare("") == 0 ) {

               /* description name */
               _description = *new std::string( getName() );
               _description.append("-");

               /* adding device type */
               _description.append( _pe->getDeviceType().getName() );
               _description.append("-");

               /* adding global id */
               char id[5]; sprintf(id, "%d", getId() );
               _description.append( id );
            }

            return _description;
         }
   };

   extern __thread BaseThread *myThread;

   BaseThread * getMyThreadSafe();

}

#endif
