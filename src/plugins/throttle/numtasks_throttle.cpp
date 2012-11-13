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
            bool _switch;
            MultipleSyncCond<LessOrEqualConditionChecker<int> > _syncCond;

            NumTasksThrottle ( const NumTasksThrottle & );
            const NumTasksThrottle & operator= ( const NumTasksThrottle & );
            
         public:
            //must be public: used in the plugin
            static const int _defaultLimit;

            NumTasksThrottle( int actualLimit = _defaultLimit, int lowerLimit = (_defaultLimit/2), bool sw = false) : _limit( actualLimit ), _lower( lowerLimit ), _switch(sw),
                                                                  _syncCond( LessOrEqualConditionChecker<int>(sys.getSchedulerStats().getTotalTasksAddr(), _lower ))
            {
#if 0
               // Debug Information
               fprintf(stderr,"Number of task throttling policy\n");
               fprintf(stderr,"minimun = %d\n", _lower);
               fprintf(stderr,"limit = %d\n", _limit);
               fprintf(stderr,"switching = %s\n", _switch?"true":"false" );
#endif

            }

            /* FIXME: disabling changing these values during execution */
            /* void setLimit( int mc ) { _limit = mc; } */

            bool throttleIn( void );
            void throttleOut ( void );

            ~NumTasksThrottle() {}
      };

      const int NumTasksThrottle::_defaultLimit = 500;

      bool NumTasksThrottle::throttleIn ( void )
      {
         if ( (sys.getTaskNum() > ( _limit * sys.getNumWorkers())) && ( sys.getReadyNum() > (sys.getNumWorkers()*2)  ))  {
            if ( _switch ) {
               _syncCond.wait();
               return true;
            } else {
               return false;
            }
         }
         else return true;
      }
      void NumTasksThrottle::throttleOut ( void )
      {
         if ( (sys.getTaskNum() < (_lower * sys.getNumWorkers()) ) && ( sys.getReadyNum() < (sys.getNumWorkers()*2)  )) {
            _syncCond.signal();
         }
      }

      //factory
      static NumTasksThrottle * createNumTasksThrottle( int actualLimit, int lowerLimit, bool sw )
      {
         return NEW NumTasksThrottle( actualLimit, lowerLimit, sw );
      }


      class NumTasksThrottlePlugin : public Plugin
      {
         private:
            int _actualLimit;
            int _lowerLimit;
            bool _switch;

         public:
            NumTasksThrottlePlugin() : Plugin( "Number of Tasks Throttle Plugin",1 ), _actualLimit( NumTasksThrottle::_defaultLimit ),
                                       _lowerLimit ( NumTasksThrottle::_defaultLimit/2), _switch(false) {}

            virtual void config( Config &cfg )
            {
               cfg.setOptionsSection( "Num tasks throttle", "Scheduling throttle policy based on the number of tasks" );

               cfg.registerConfigOption ( "throttle-limit", NEW Config::PositiveVar( _actualLimit ),
                  "Defines the number of tasks per thread allowed" );
               cfg.registerArgOption ( "throttle-limit", "throttle-limit" );

               cfg.registerConfigOption ( "throttle-min", NEW Config::PositiveVar( _lowerLimit ),
                  "Defines the minumun number of tasks per thread to re-active task creation" );
               cfg.registerArgOption ( "throttle-min", "throttle-min" );

               cfg.registerConfigOption ( "throttle-switch", NEW Config::FlagOption( _switch ),
                  "Defines the behaviour for throttling policy" );
               cfg.registerArgOption ( "throttle-switch", "throttle-switch" );
            }

            virtual void init() {
               sys.setThrottlePolicy( createNumTasksThrottle( _actualLimit, _lowerLimit, _switch )); 
            }
      };

   }
}

DECLARE_PLUGIN("throttle-numtasks",nanos::ext::NumTasksThrottlePlugin);
