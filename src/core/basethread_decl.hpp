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

#ifndef _BASE_THREAD_DECL
#define _BASE_THREAD_DECL

#include "workdescriptor_fwd.hpp"
#include "processingelement_fwd.hpp"
#include "debug.hpp"
#include "atomic_decl.hpp"
#include "schedule_fwd.hpp"
#include "threadteam_fwd.hpp"
#include "allocator_decl.hpp"

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
         unsigned          _id;
         ThreadTeam       *_team;
         unsigned          _singleCount;
         SchedData        *_schedData;
         TeamData         *_parentData;
         nanos_ws_desc_t  *_wsDescriptor; /*< pointer to last worksharing descriptor */
         bool              _star;         /*< is current thread a star (or role) within the team */

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
         TeamData () : _id( 0 ), _team(NULL), _singleCount( 0 ), _schedData( NULL ), _parentData( NULL ), _wsDescriptor( NULL ), _star( false ) {}
        /*! \brief TeamData destructor
         */
         ~TeamData ();

         unsigned getId() const { return _id; }
         ThreadTeam *getTeam() const { return _team; }
         unsigned getSingleCount() const { return _singleCount; }

         void setId ( int id ) { _id = id; }
         void setTeam ( ThreadTeam *team ) { _team = team; }
         unsigned nextSingleGuard() {
            ++_singleCount;
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
         nanos_ws_desc_t *getTeamWorkSharingDescriptor( bool *b );

         void setParentTeamData ( TeamData *data ) { _parentData = data; }
         TeamData * getParentTeamData () const { return _parentData; }
   };

   namespace ext {
   class SMPMultiThread;
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
         ext::SMPMultiThread *   _parent;

         ProcessingElement *     _pe;         /**< Threads are binded to a PE for its life-time */
         WD &                    _threadWD;

         // Thread status
         bool                    _started;
         volatile bool           _mustStop;
         volatile bool           _paused;
         WD *                    _currentWD;
         WD *                    _nextWD;

         // Team info
         bool                    _hasTeam;
         TeamData *              _teamData;
         TeamData *              _nextTeamData; /**< Next team data, thread is already registered in a new team but has not enter yet */

         nanos_ws_desc_t         _wsDescriptor; /**< Local worksharing descriptor */

         Allocator               _allocator;

         virtual void initializeDependent () = 0;
         virtual void runDependent () = 0;

         // These must be called through the Scheduler interface
         virtual void switchHelperDependent( WD* oldWD, WD* newWD, void *arg ) = 0;
         virtual void exitHelperDependent( WD* oldWD, WD* newWD, void *arg ) = 0;
         virtual void inlineWorkDependent (WD &work) = 0;
         virtual void outlineWorkDependent (WD &work) = 0;
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
         BaseThread ( WD &wd, ProcessingElement *creator=0, ext::SMPMultiThread *parent=NULL ) :
               _id( _idSeed++ ), _name("Thread"), _description(""), _parent( parent ), _pe( creator ), _threadWD( wd ),
               _started( false ), _mustStop( false ), _paused( false ), _currentWD( NULL),
               _nextWD( NULL), _hasTeam( false ), _teamData(NULL), _nextTeamData(NULL), _allocator() {}

        /*! \brief BaseThread destructor
         */
         virtual ~BaseThread()
         {
            ensure0(!_hasTeam,"Destroying thread inside a team!");
            ensure0(!_started,"Trying to destroy running thread");
         }


         // atomic access
         void lock ();

         void unlock ();

         virtual void start () = 0;
         void run();
         void stop();
         
         void pause ();
         void unpause ();

         virtual void idle() {};
         virtual void yield() {};

         virtual void join() = 0;
         virtual void bind() {};

         // set/get methods
         void setCurrentWD ( WD &current );

         WD * getCurrentWD () const;

         WD & getThreadWD () const;

         void resetNextWD ();
         bool setNextWD ( WD *next );

         bool reserveNextWD ( void );
         bool setReservedNextWD ( WD *next );

         WD * getNextWD () const;

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

         void setNextTeamData( TeamData *td);

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
         
         //! \brief Is the thread paused as the result of stopping the scheduler?
         bool isPaused () const;

         ProcessingElement * runningOn() const;

         void associate();

         int getId();

         int getCpuId();

         bool singleGuard();

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
         virtual void notifyOutlinedCompletionDependent( WD &completedWD ) {
            fatal0( "::notifyOutlinedCompletionDependent() not available for this thread type." );
         }
         virtual bool isCluster() = 0;
   };

   extern __thread BaseThread *myThread;

   BaseThread * getMyThreadSafe();

}

#endif
