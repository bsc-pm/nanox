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

#ifndef _NANOS_SCHEDULE
#define _NANOS_SCHEDULE

#include <string>

#include "workdescriptor.hpp"
#include "wddeque.hpp"
#include "basethread.hpp"
#include "atomic.hpp"
#include "functors.hpp"
#include <algorithm>

namespace nanos
{

   class SchedulingData
   {

      private:
         int schId;

         SchedulingData ( const SchedulingData & );
         const SchedulingData & operator= ( const SchedulingData & );

      public:

         // constructor
         SchedulingData( int id=0 ) : schId( id ) {}

         // destructor
         ~SchedulingData() {}

         void setSchId( int id )  { schId = id; }

         int getSchId() const { return schId; }
   };

// Groups a number of BaseThreads and a number of WD with a policy
// Each BaseThread and WD can pertain only to a SG

   class SchedulingGroup
   {

      private:
         typedef std::vector<SchedulingData *> Group;

         std::string    _name;
         WDDeque        _idleQueue;

         Group          _group;

         // disable copy and assignment
         SchedulingGroup( const SchedulingGroup & );
         SchedulingGroup & operator= ( const SchedulingGroup & );

         void init( int groupSize );

      public:
         // constructors
         SchedulingGroup( std::string &policy_name, int groupSize=1 ) : _name( policy_name ) { init( groupSize ); }
         SchedulingGroup( const char  *policy_name, int groupSize=1 ) : _name( policy_name ) { init( groupSize ); }

         // destructor
         virtual ~SchedulingGroup()
         {
             std::for_each( _group.begin(),_group.end(), deleter<SchedulingData> );
         }

         //modifiers
         SchedulingData * getMemberData( int id ) { return _group[id]; }

         int getSize() { return _group.size(); }

         // membership related methods. This members are not thread-safe
         virtual void addMember ( BaseThread &thread );
         virtual void removeMember ( BaseThread &thread );
         virtual SchedulingData * createMemberData ( BaseThread &thread ) { return new SchedulingData(); };

         // policy related methods
         virtual WD *atCreation ( BaseThread *thread, WD &newWD ) { return 0; }

         virtual WD *atIdle     ( BaseThread *thread ) = 0;
         virtual WD *atExit     ( BaseThread *thread ) { return atIdle( thread ); }

         virtual WD *atBlock    ( BaseThread *thread, WD *hint=0 ) { return atIdle( thread ); }

         virtual WD *atWakeUp   ( BaseThread *thread, WD &wd ) { return 0; }

         virtual void queue ( BaseThread *thread,WD &wd )  = 0;

         // idle management
         virtual void queueIdle ( BaseThread *thread,WD &wd );
   };

// singleton class to encapsulate scheduling data and methods

   class GenericSyncCond;
   typedef void SchedulerHelper ( WD *oldWD, WD *newWD, void *arg);

   class Scheduler
   {
      public:
         static void inlineWork ( WD *work );
         static void switchHelper (WD *oldWD, WD *newWD, void *arg);
         static void exitHelper (WD *oldWD, WD *newWD, void *arg);

         static void submit ( WD &wd );
         static void exit ( void );
         static void switchTo ( WD *to );
         static void exitTo ( WD *next );

         static void idle ( void );
         static void queue ( WD &wd );
         static void yield ();

         static void switchToThread ( BaseThread * thread );

         static void waitOnCondition ( GenericSyncCond *condition );
         static void wakeUp ( WD *wd );
   };

   typedef SchedulingGroup SG;
   typedef SG * ( *sgFactory ) ( int groupSize );

   class SchedulePolicy
   {
      public:

         virtual ~SchedulePolicy ();

         virtual WD * atSubmit (BaseThread *thread, WD &wd);

         virtual WD *atIdle     ( BaseThread *thread ) = 0;
         virtual WD *atExit     ( BaseThread *thread, WD *current ) { return atIdle( thread ); }
         virtual WD *atBlock    ( BaseThread *thread, WD *current ) { return atIdle( thread ); }
         virtual WD *atYield    ( BaseThread *thread, WD *current) { return atIdle(thread); };
         virtual WD *atWakeUp   ( BaseThread *thread, WD &wd ) { return 0; }

         virtual void queue ( BaseThread *thread, WD &wd )  = 0;
   };
   
};

#endif

