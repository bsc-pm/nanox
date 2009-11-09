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

namespace nanos {
namespace ext {

//TODO: only works with 1 scheduling group

class IdleThreadsThrottle: public ThrottlePolicy
{

   private:
      int _limit;
      static const int _defaultLimit;

   public:
      IdleThreadsThrottle() : _limit(_defaultLimit) {}

      void init() {}

      void setMaxCutoff( int mi ) { _limit = mi; }

      bool throttle();

      ~IdleThreadsThrottle() {}
};

const int IdleThreadsThrottle::_defaultLimit = 0;


bool IdleThreadsThrottle::throttle()
{
   //checking if the number of idle tasks is higher than the allowed maximum
   if ( sys.getIdleNum() > _limit )  {
      verbose0( "Cutoff Policy: avoiding task creation!" );
      return false;
   }

   return true;
}

//factory
static IdleThreadsThrottle * createIdleThrottle()
{
   return new IdleThreadsThrottle();
}


class IdleThreadsThrottlePlugin : public Plugin
{

   public:
      IdleThreadsThrottlePlugin() : Plugin( "Idle Threads Throttling Plugin",1 ) {}

      virtual void init() {
         sys.setThrottlePolicy( createIdleThrottle() );
      }
};

}}

nanos::ext::IdleThreadsThrottlePlugin NanosXPlugin;