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

#include "system.hpp"
#include "throttle.hpp"
#include "plugin.hpp"
#include "config.hpp"


namespace nanos {
   namespace ext {

      class NumTasksThrottle: public ThrottlePolicy
      {

         private:
            int _limit;

            NumTasksThrottle ( const NumTasksThrottle & );
            const NumTasksThrottle & operator= ( const NumTasksThrottle & );
            
         public:
            //must be public: used in the plugin
            static const int _defaultLimit;

            NumTasksThrottle( int actualLimit = _defaultLimit ) : _limit( actualLimit ) {}

            void setLimit( int mc ) { _limit = mc; }

            bool throttle();

            ~NumTasksThrottle() {}
      };

      const int NumTasksThrottle::_defaultLimit = 5;

      bool NumTasksThrottle::throttle()
      {
         if ( sys.getTaskNum() > _limit*sys.getNumWorkers() ) {
            return false;
         }

         return true;
      }

      //factory
      static NumTasksThrottle * createNumTasksThrottle( int actualLimit )
      {
         return new NumTasksThrottle( actualLimit );
      }


      class NumTasksThrottlePlugin : public Plugin
      {

         public:
            NumTasksThrottlePlugin() : Plugin( "Number of Tasks Throttle Plugin",1 ) {}

            virtual void init() {
               Config config;

               int actualLimit = NumTasksThrottle::_defaultLimit; 
               config.registerArgOption( new Config::PositiveVar( "throttle-limit",
                                          actualLimit ) );
               config.init(); 
               sys.setThrottlePolicy( createNumTasksThrottle( actualLimit )); 
            }
      };

   }
}

nanos::ext::NumTasksThrottlePlugin NanosXPlugin;
