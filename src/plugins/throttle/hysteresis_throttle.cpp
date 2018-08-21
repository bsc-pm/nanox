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

#include "throttle_decl.hpp"
#include "system.hpp"
#include "plugin.hpp"
#include "config.hpp"


namespace nanos {
   namespace ext {

      class HysteresisThrottle: public ThrottlePolicy
      {
         private:
            typedef int (*ntask_getter_t)( void ) ;
            static int get_total_tasks (void) { return sys.getTaskNum(); }
            static int get_ready_tasks (void) { return sys.getReadyNum(); }
         private:
            int                                                  _upper;
            int                                                  _lower;
            std::string                                          _type;
            MultipleSyncCond<LessOrEqualConditionChecker<int> > *_syncCond;
            ntask_getter_t                                       _get_num_tasks;

            HysteresisThrottle ( const HysteresisThrottle & );
            const HysteresisThrottle & operator= ( const HysteresisThrottle & );

         public:
            HysteresisThrottle( int upper, int lower, std::string type )
            :  _upper( upper * sys.getNumThreads() ),
               _lower( lower * sys.getNumThreads() ),
               _type ( type ),
               _syncCond( NULL )
            {
               if ( _type == "total" ) {
                  _syncCond = (MultipleSyncCond< LessOrEqualConditionChecker<int> >*) new MultipleSyncCond< LessOrEqualConditionChecker<int> >(LessOrEqualConditionChecker<int>(sys.getSchedulerStats().getTotalTasksAddr(), lower * sys.getNumThreads())) ;
                  _get_num_tasks = &get_total_tasks;
               } else if ( _type == "ready" ) {
                  _syncCond = (MultipleSyncCond< LessOrEqualConditionChecker<int> >*) new MultipleSyncCond< LessOrEqualConditionChecker<int> >(LessOrEqualConditionChecker<int>(sys.getSchedulerStats().getReadyTasksAddr(), lower * sys.getNumThreads())) ;
                  _get_num_tasks = &get_ready_tasks;
               } else fatal0("Unknow throttle type");

               verbose0( "Throttle hysteresis created");
               verbose0( "   type of tasks: " << _type );
               verbose0( "   lower bound: " << lower * sys.getNumThreads() );
               verbose0( "   upper bound: " << upper * sys.getNumThreads() );
            }

            void setUpper( int upper ) { _upper = upper; };
            void setLower( int lower ) { _lower = lower; };

            bool throttleIn( void );
            void throttleOut ( void );

            ~HysteresisThrottle() {
               delete _syncCond;
            }
      };

      bool HysteresisThrottle::throttleIn ( void )
      {
         // If it's OpenMP, first level tasks will have depth 1
         unsigned maxDepth = ( sys.getPMInterface().getInterface() == PMInterface::OpenMP ) ? 2 : 1;
         // Only dealing with first level tasks
         if ( ( (myThread->getCurrentWD())->getDepth() < maxDepth ) && ( _get_num_tasks() > _upper ) ) _syncCond->wait();
         return true;
      }
      void HysteresisThrottle::throttleOut ( void )
      {
         if ( _get_num_tasks() <= _lower ) _syncCond->signal();
      }

      class HysteresisThrottlePlugin : public Plugin
      {
         private:
            int         _lowerLimit;
            int         _upperLimit;
            std::string _type;

         public:
            HysteresisThrottlePlugin() : Plugin( "Hysteresis throttle plugin (Hysteresis in number of tasks per thread)",1 ),
                                      _lowerLimit( 250 ), _upperLimit ( 500 ), _type("total") {}

            virtual void config( Config &cfg )
            {
               cfg.setOptionsSection( "Hysteresis throttle", "Scheduling throttle policy based on the number of tasks" );

               cfg.registerConfigOption ( "throttle-upper", NEW Config::PositiveVar( _upperLimit ),
                  "Defines the maximum number of tasks (per thread) allowed to create new 1st level's tasks (500 * nthreads)" );
               cfg.registerArgOption ( "throttle-upper", "throttle-upper" );

               cfg.registerConfigOption ( "throttle-lower", NEW Config::PositiveVar( _lowerLimit ),
                  "Defines the number of tasks (per thread) to re-active 1st level task creation (250 * nthreads)" );
               cfg.registerArgOption ( "throttle-lower", "throttle-lower" );

               cfg.registerConfigOption ( "throttle-type", NEW Config::StringVar( _type ),
                  "Defines the task's type (ready or total) we have to take into account to stop or re-active 1st level task creation (total)" );
               cfg.registerArgOption ( "throttle-type", "throttle-type" );

            }

            virtual void init() {
               sys.setThrottlePolicy( NEW HysteresisThrottle( _upperLimit, _lowerLimit, _type ) ); 
            }
      };

   }
}

DECLARE_PLUGIN("throttle-default",nanos::ext::HysteresisThrottlePlugin);
