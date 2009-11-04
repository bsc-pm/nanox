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

class TaskDepthThrottle: public ThrottlePolicy
{

   private:
      int _limit;
      static const int _defaultLimit;

   public:
      TaskDepthThrottle() : _limit( _defaultLimit ) {}

      void setLimit( int ml ) { _limit = ml; }

      bool throttle();

      ~TaskDepthThrottle() {}
};

const int _defaultLimit = 4;

bool TaskDepthThrottle::throttle()
{
   //checking the parent level of the next work to be created (check >)
   if ( ( myThread->getCurrentWD() )->getLevel() > _limit )  {
      return false;
   }

   return true;
}

//factory
static TaskDepthThrottle * createTaskDepthThrottle()
{
   return new TaskDepthThrottle();
}

class TaskDepthThrottlePlugin : public Plugin
{

   public:
      TaskDepthThrottlePlugin() : Plugin( "Task Tree Level CutOff Plugin",1 ) {}

      virtual void init() {
         sys.setThrottlePolicy( createTaskDepthThrottle() );
      }
};

}}

nanos::ext::TaskDepthThrottlePlugin NanosXPlugin;
