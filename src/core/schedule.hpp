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

namespace nanos
{

   class SchedulingData
   {

      private:
         int schId;

      public:

         // constructor
         SchedulingData( int id=0 ) : schId( id ) {}

         //TODO: copy & assigment costructor

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
         typedef std::vector<SchedulingData *> group_t;

         std::string    name;
         WDDeque        idleQueue;

         group_t        group;

         // disable copy and assignment
         SchedulingGroup( const SchedulingGroup & );
         SchedulingGroup & operator= ( const SchedulingGroup & );

         void init( int groupSize );

      public:
         // constructors
         SchedulingGroup( std::string &policy_name, int groupSize=1 ) : name( policy_name ) { init( groupSize ); }

         SchedulingGroup( const char  *policy_name, int groupSize=1 ) : name( policy_name ) { init( groupSize ); }

         // destructor
         virtual ~SchedulingGroup() {}


         //modifiers
         SchedulingData * getMemberData( int id ) { return group[id]; }

         int getSize() { return group.size(); }

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
         virtual WD *getIdle( BaseThread *thread );

   };

// singleton class to encapsulate scheduling data and methods

   class Scheduler
   {

      public:
         static void submit ( WD &wd );
         static void exit ( void );

         template<typename T>
         static void blockOnCondition ( volatile T *var, T condition = 0 );
         template<typename T>
         static void blockOnConditionLess ( volatile T *var, T condition = 0 );

         static void idle ( void );
         static void queue ( WD &wd );
   };

   typedef SchedulingGroup SG;
   typedef SG * ( *sgFactory ) ( int groupSize );

};

#endif

