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

namespace nanos {
   namespace ext {

      class BreadthFirstPolicy : public SchedulingGroup
      {

         private:
            WDDeque           _readyQueue;

         public:
            static bool      _useStack;

         public:
            // constructor
            BreadthFirstPolicy( int groupSize ) : SchedulingGroup( "breadth-first-sch",groupSize ) {}

            // TODO: copy and assigment operations
            // destructor
            virtual ~BreadthFirstPolicy() {}

            virtual WD *atCreation ( BaseThread *thread, WD &newWD );
          virtual WD *atIdle ( BaseThread *thread );
            virtual void queue ( BaseThread *thread, WD &wd );
      };

      void BreadthFirstPolicy::queue ( BaseThread *thread, WD &wd )
      {
         _readyQueue.push_back( &wd );
      }

      WD * BreadthFirstPolicy::atCreation ( BaseThread *thread, WD &newWD )
      {
         queue( thread,newWD );
         return 0;
      }

      WD * BreadthFirstPolicy::atIdle ( BaseThread *thread )
      {
         if ( _useStack ) return _readyQueue.pop_back( thread );

         return _readyQueue.pop_front( thread );
      }

      bool BreadthFirstPolicy::_useStack = false;

      // Factory
      static SchedulingGroup * createBreadthFirstPolicy ( int groupSize )
      {
         return new BreadthFirstPolicy( groupSize );
      }

      class BFSchedPlugin : public Plugin
      {

         public:
            BFSchedPlugin() : Plugin( "BF scheduling Plugin",1 ) {}

            virtual void init() {
               Config config;

               config.registerArgOption( new Config::FlagOption( "nth-bf-use-stack",BreadthFirstPolicy::_useStack ) );
               config.registerArgOption( new Config::FlagOption( "nth-bf-stack",BreadthFirstPolicy::_useStack ) );
               config.init();

               sys.setDefaultSGFactory( createBreadthFirstPolicy );
            }
      };

   }
}

nanos::ext::BFSchedPlugin NanosXPlugin;

