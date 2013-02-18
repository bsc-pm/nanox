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

#include "throttle_decl.hpp"
#include "system.hpp"
#include "plugin.hpp"
#include "config.hpp"


namespace nanos {
   namespace ext {

      class HysteresisThrottle: public ThrottlePolicy
      {
         private:
            int   _upper;
            int   _lower;
            MultipleSyncCond<LessOrEqualConditionChecker<int> > _syncCond;

            HysteresisThrottle ( const HysteresisThrottle & );
            const HysteresisThrottle & operator= ( const HysteresisThrottle & );

         public:
            HysteresisThrottle( int upper, int lower ) : _upper( upper ), _lower( lower ),
                                                      _syncCond( LessOrEqualConditionChecker<int>(sys.getSchedulerStats().getTotalTasksAddr(), lower )) {}

            void setUpper( int upper ) { _upper = upper; };
            void setLower( int lower ) { _lower = lower; };

            bool throttleIn( void );
            void throttleOut ( void );

            ~HysteresisThrottle() {}
      };

      bool HysteresisThrottle::throttleIn ( void )
      {
         // Only dealing with first level tasks
         if ( ( (myThread->getCurrentWD())->getDepth() < 1 ) && ( sys.getTaskNum() > (_upper * sys.getNumWorkers()) ) ) _syncCond.wait();
         return true;
      }
      void HysteresisThrottle::throttleOut ( void )
      {
         if ( sys.getTaskNum() <= ( _lower * sys.getNumWorkers() ) ) _syncCond.signal();
      }

      class HysteresisThrottlePlugin : public Plugin
      {
         private:
            int _lowerLimit;
            int _upperLimit;

         public:
            HysteresisThrottlePlugin() : Plugin( "Hysteresis throttle plugin (Hysteresis in number of tasks per thread)",1 ),
                                      _lowerLimit( 250 ), _upperLimit ( 500 ) {}

            virtual void config( Config &cfg )
            {
               cfg.setOptionsSection( "Hysteresis throttle (hysteresis in number of tasks per thread)", "Scheduling throttle policy based on the number of tasks" );

               cfg.registerConfigOption ( "throttle-upper", NEW Config::PositiveVar( _upperLimit ),
                  "Defines the maximum number of tasks (per thread) allowed to create new 1st level's tasks" );
               cfg.registerArgOption ( "throttle-upper", "throttle-upper" );

               cfg.registerConfigOption ( "throttle-lower", NEW Config::PositiveVar( _lowerLimit ),
                  "Defines the number of tasks (per thread) to re-active 1st level task creation" );
               cfg.registerArgOption ( "throttle-lower", "throttle-lower" );

            }

            virtual void init() {
               sys.setThrottlePolicy( NEW HysteresisThrottle( _upperLimit, _lowerLimit ) ); 
            }
      };

   }
}

DECLARE_PLUGIN("throttle-default",nanos::ext::HysteresisThrottlePlugin);
