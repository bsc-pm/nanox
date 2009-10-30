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

#include "schedule.hpp"
#include "wddeque.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "config.hpp"

using namespace nanos;

class BreadthFirstPolicy : public SchedulingGroup
{

   private:
      WDDeque   readyQueue;
      bool      useStack;

   public:
      // constructor
      BreadthFirstPolicy( bool stack, int groupSize ) : SchedulingGroup( "breadth-first-sch",groupSize ), useStack( stack ) {}

      // TODO: copy and assigment operations
      // destructor
      virtual ~BreadthFirstPolicy() {}

      virtual WD *atCreation ( BaseThread *thread, WD &newWD );
      virtual WD *atIdle ( BaseThread *thread );
      virtual void queue ( BaseThread *thread, WD &wd );
};

void BreadthFirstPolicy::queue ( BaseThread *thread, WD &wd )
{
   readyQueue.push_back( &wd );
}

WD * BreadthFirstPolicy::atCreation ( BaseThread *thread, WD &newWD )
{
   queue( thread,newWD );
   return 0;
}

WD * BreadthFirstPolicy::atIdle ( BaseThread *thread )
{
   if ( useStack ) return readyQueue.pop_back( thread );

   return readyQueue.pop_front( thread );
}

static bool useStack = false;

// Factory
SchedulingGroup * createBreadthFirstPolicy ( int groupSize )
{
   return new BreadthFirstPolicy( useStack,groupSize );
}

class BFSchedPlugin : public Plugin
{

   public:
      BFSchedPlugin() : Plugin( "BF scheduling Plugin",1 ) {}

      virtual void init() {
         Config config;

         config.registerArgOption( new Config::FlagOption( "nth-bf-use-stack",useStack ) );
         config.registerArgOption( new Config::FlagOption( "nth-bf-stack",useStack ) );
         config.init();

         sys.setDefaultSGFactory( createBreadthFirstPolicy );
      }
};

BFSchedPlugin NanosXPlugin;

