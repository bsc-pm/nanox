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

      class NumTasksThrottle: public ThrottlePolicy
      {

         private:
            int _limit;
            int _lower;
            MultipleSyncCond<LessOrEqualConditionChecker<int> > _syncCond;

            NumTasksThrottle ( const NumTasksThrottle & );
            const NumTasksThrottle & operator= ( const NumTasksThrottle & );
            
         public:
            //must be public: used in the plugin
            static const int _defaultLimit;

            NumTasksThrottle( int actualLimit = _defaultLimit, int lowerLimit = (_defaultLimit/2) ) : _limit( actualLimit ), _lower( lowerLimit ),
                                                                  _syncCond( LessOrEqualConditionChecker<int>(sys.getSchedulerStats().getTotalTasksAddr(), _lower )) {}

            /* FIXME: disabling changing these values during execution */
            /* void setLimit( int mc ) { _limit = mc; } */

            bool throttleIn( void );
            void throttleOut ( void );

            ~NumTasksThrottle() {}
      };

      const int NumTasksThrottle::_defaultLimit = 500;

      bool NumTasksThrottle::throttleIn ( void )
      {
         if ( sys.getTaskNum() > (_limit * sys.getNumWorkers()) ) {
            _syncCond.wait();
         }
         return true;
      }
      void NumTasksThrottle::throttleOut ( void )
      {
         if ( sys.getTaskNum() < (_lower * sys.getNumWorkers()) ) {
            _syncCond.signal();
         }
      }

      //factory
      static NumTasksThrottle * createNumTasksThrottle( int actualLimit, int lowerLimit )
      {
         return NEW NumTasksThrottle( actualLimit, lowerLimit );
      }


      class NumTasksThrottlePlugin : public Plugin
      {
         private:
            int _actualLimit;
            int _lowerLimit;

         public:
            NumTasksThrottlePlugin() : Plugin( "Number of Tasks Throttle Plugin",1 ), _actualLimit( NumTasksThrottle::_defaultLimit ) {}

            virtual void config( Config &cfg )
            {
               cfg.setOptionsSection( "Num tasks throttle", "Scheduling throttle policy based on the number of tasks" );

               cfg.registerConfigOption ( "throttle-limit", NEW Config::PositiveVar( _actualLimit ),
                  "Defines the number of tasks per thread allowed" );
               cfg.registerArgOption ( "throttle-limit", "throttle-limit" );

               cfg.registerConfigOption ( "throttle-min", NEW Config::PositiveVar( _lowerLimit ),
                  "Defines the minumun number of tasks per thread to re-active task creation" );
               cfg.registerArgOption ( "throttle-min", "throttle-min" );
            }

            virtual void init() {
               sys.setThrottlePolicy( createNumTasksThrottle( _actualLimit, _lowerLimit )); 
            }
      };

   }
}

DECLARE_PLUGIN("throttle-numtasks",nanos::ext::NumTasksThrottlePlugin);
