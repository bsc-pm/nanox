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

#define DEFAULT_CUTOFF_IDLE 5

namespace nanos {
namespace ext {

//TODO: only works with 1 scheduling group

class idleThrottle: public ThrottlePolicy
{

   private:
      int _maxIdle;

   public:
      idleThrottle() : _maxIdle( DEFAULT_CUTOFF_IDLE ) {}

      void init() {}

      void setMaxCutoff( int mi ) { _maxIdle = mi; }

      bool throttle();

      ~idleThrottle() {}
};


bool idleThrottle::throttle()
{
   //checking if the number of idle tasks is higher than the allowed maximum
   if ( sys.getIdleNum() > _maxIdle )  {
      verbose0( "Cutoff Policy: avoiding task creation!" );
      return false;
   }

   return true;
}

//factory
ThrottlePolicy * createIdleThrottle()
{
   return new idleThrottle();
}


class IdleThrottlePlugin : public Plugin
{

   public:
      IdleThrottlePlugin() : Plugin( "Idle Threads Throttling Plugin",1 ) {}

      virtual void init() {
         sys.setCutOffPolicy( createIdleThrottle() );
      }
};

}}

nanos::ext::IdleThrottlePlugin NanosXPlugin;
